"""
LTC2265-14 ADC High Level Analyzer — 2-Lane, 14-Bit Serialization
===================================================================
Input: Simple Parallel (SP) analyzer
  SimpleParallel D0    = OUT1A  (odd  bits: D13,D11,...,D1)
  SimpleParallel D1    = OUT1B  (even bits: D12,D10,...,D0)
  SimpleParallel Clock = DCO    (BOTH edges — set "Clock Edge" to "Both" in SimpleParallel settings)

The Simple Parallel analyzer emits one frame per DCO clock edge:
  frame.type        == "data"
  frame.data["data"] == bytes (little-endian), 1 byte for a 2-bit bus
                        e.g. b'\\x01' means D0=1, D1=0

Each ADC sample requires exactly 7 SP frames (7 DCO edges):
  Edge 1: A=D13, B=D12
  Edge 2: A=D11, B=D10
  ...
  Edge 7: A=D1,  B=D0

Alignment: trigger Logic 2 on FRAME rising edge so capture starts at edge 1.
Use "Bit Offset" setting if the trigger fires slightly late.
"""

from saleae.analyzers import HighLevelAnalyzer, AnalyzerFrame, NumberSetting, ChoicesSetting

# Number of DCO edges (SP frames) per ADC sample in 2-lane 14-bit mode
BITS_PER_FRAME = 7


class LTC2265Decoder(HighLevelAnalyzer):
    """LTC2265-14 — 2-lane, 14-bit serialization, input from Simple Parallel.

    Simple Parallel LLA frame format (frame.type == "data"):
        frame.data["data"] — bytes object, little-endian, containing the
                             sampled parallel word at each DCO edge.
        bit 0 of word = SP D0 = OUT1A (odd  ADC bits: D13,D11,...,D1)
        bit 1 of word = SP D1 = OUT1B (even ADC bits: D12,D10,...,D0)
    """

    # ------------------------------------------------------------------ #
    # Settings visible to the user in Logic 2
    # ------------------------------------------------------------------ #

    output_format = ChoicesSetting(
        label="Output Format",
        choices=(
            "Signed Decimal (2s complement)",
            "Unsigned Decimal",
            "Hex",
            "Voltage (approx)",
        )
    )

    vref_mv = NumberSetting(
        label="Full-Scale Range mVpp (Voltage mode only)",
        min_value=500,
        max_value=2100
    )

    bit_offset = NumberSetting(
        label="Bit Offset (0-6): skip N SP frames before first sample",
        min_value=0,
        max_value=6
    )

    # ------------------------------------------------------------------ #
    # Frame types shown in the Logic 2 protocol results table
    # ------------------------------------------------------------------ #

    result_types = {
        "sample": {
            "format": "ADC: {{data.value}}  raw=0x{{data.raw}}"
        },
    }

    # ------------------------------------------------------------------ #

    def __init__(self):
        # NOTE: Settings (self.bit_offset etc.) ARE available here in the
        # modern Logic 2 HLA API — they are class-level descriptors set
        # before __init__ is called.
        self._edge_count  = 0
        self._lane_a      = 0   # 7-bit shift register for OUT1A
        self._lane_b      = 0   # 7-bit shift register for OUT1B
        self._group_start = None
        self._skip_remaining = int(self.bit_offset)
        # Debug flag — set True to print every incoming frame to the terminal
        self._debug = True
        self._debug_printed = False  # only print first frame once

    # ------------------------------------------------------------------ #
    # Value formatting
    # ------------------------------------------------------------------ #

    def _format(self, adc14):
        """adc14 is unsigned 0..16383. Returns display string."""
        fmt = self.output_format

        if fmt == "Hex":
            return f"0x{adc14:04X}"

        if fmt == "Unsigned Decimal":
            return str(adc14)

        if fmt == "Voltage (approx)":
            fsr = float(self.vref_mv)
            mv = (adc14 - 8192) * fsr / 16384.0
            return f"{mv:.2f} mV"

        # Default: signed 2's complement
        signed = adc14 if adc14 < 8192 else adc14 - 16384
        return str(signed)

    # ------------------------------------------------------------------ #
    # Reconstruct 14-bit value from two 7-bit shift registers
    # ------------------------------------------------------------------ #

    def _reconstruct(self):
        """
        After 7 edges the shift registers contain (MSB shifted in first):
          _lane_a: bit6=D13, bit5=D11, bit4=D9, bit3=D7, bit2=D5, bit1=D3, bit0=D1
          _lane_b: bit6=D12, bit5=D10, bit4=D8, bit3=D6, bit2=D4, bit1=D2, bit0=D0
        Interleave into 14-bit word [D13..D0].
        """
        adc14 = 0
        for i in range(7):
            a_bit = (self._lane_a >> (6 - i)) & 1
            b_bit = (self._lane_b >> (6 - i)) & 1
            adc14 |= a_bit << (13 - 2 * i)
            adc14 |= b_bit << (12 - 2 * i)
        return adc14 & 0x3FFF

    # ------------------------------------------------------------------ #
    # Main decode — called once per SP frame (= one DCO edge)
    # ------------------------------------------------------------------ #

    def decode(self, frame: AnalyzerFrame):
        """
        Simple Parallel delivers one frame per DCO clock edge.

        Expected:
          frame.type        == "data"
          frame.data["data"] == bytes object, e.g. b'\\x03' for D0=1, D1=1
          frame.start_time  == timestamp of the DCO edge
          frame.end_time    == timestamp of the next DCO edge

        The parallel word is decoded as a little-endian integer:
          word = int.from_bytes(frame.data["data"], "little")
          out1a = word & 1        (SP D0)
          out1b = (word >> 1) & 1 (SP D1)
        """

        # ---- debug: print the very first frame to the Logic 2 terminal ----
        if self._debug and not self._debug_printed:
            print(f"[LTC2265] First frame: type={frame.type!r}  "
                  f"data_keys={list(frame.data.keys())}  "
                  f"data={frame.data}")
            self._debug_printed = True

        # ---- guard: Simple Parallel emits type "data" ---------------------
        # Accept both "data" and any other type defensively, but only
        # process known-good frame types.
        if frame.type != "data":
            # Log unexpected types once to help diagnose mis-wiring
            print(f"[LTC2265] Unexpected frame type: {frame.type!r} — "
                  f"expected 'data' from Simple Parallel analyzer")
            return None

        # ---- extract the 2-bit parallel word ------------------------------
        raw = frame.data.get("data", b"\x00")

        # The SP analyzer delivers data as a bytes object (little-endian).
        # Convert defensively — handle both bytes and int.
        if isinstance(raw, (bytes, bytearray)):
            word = int.from_bytes(raw, "little")
        elif isinstance(raw, int):
            word = raw
        else:
            word = 0

        out1a = (word >> 0) & 1   # SP D0 = OUT1A = odd  bits (D13,D11,...,D1)
        out1b = (word >> 1) & 1   # SP D1 = OUT1B = even bits (D12,D10,...,D0)

        # ---- skip frames for alignment (bit_offset setting) ---------------
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return None

        # ---- start of a new 7-edge group ----------------------------------
        if self._edge_count == 0:
            self._group_start = frame.start_time
            self._lane_a      = 0
            self._lane_b      = 0

        # ---- shift in this edge (MSB arrives first → shift left) ----------
        self._lane_a = ((self._lane_a << 1) | out1a) & 0x7F
        self._lane_b = ((self._lane_b << 1) | out1b) & 0x7F
        self._edge_count += 1

        # ---- after 7 edges, emit one decoded ADC sample -------------------
        if self._edge_count == BITS_PER_FRAME:
            adc14 = self._reconstruct()
            value = self._format(adc14)

            result = AnalyzerFrame(
                "sample",
                self._group_start,
                frame.end_time,
                {
                    "value": value,
                    "raw":   f"{adc14:04X}",
                }
            )

            self._edge_count = 0
            return result

        return None