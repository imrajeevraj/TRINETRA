import re

# Indian Motor Vehicles Act standard plate formats:
#   Standard : [State(2)][District(1-2)][Series(1-2)][Number(4)]  e.g. DL12AB1234
#   BH-series : [YY]BH[Number(4)][Class(1-2)]                    e.g. 22BH1234AA
_STANDARD_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
_BH_SERIES_PLATE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


class PlateNormalizer:
    @staticmethod
    def normalize(raw_text: str) -> str:
        """
        Normalize plate text:
        - Convert to uppercase
        - Remove whitespace and non-alphanumeric characters
        """
        if not raw_text:
            return ""

        # Convert to upper case
        text = raw_text.upper()

        # Remove anything that isn't a letter or number
        text = re.sub(r"[^A-Z0-9]", "", text)

        return text

    @staticmethod
    def is_valid_format(normalized_text: str) -> bool:
        """
        Check if the normalized text matches a valid Indian licence-plate format.

        Accepts:
          - Standard state-code format: ``DL12AB1234`` (2-letter state, 1-2 digit
            district, 1-3 letter series, 4 digits).
          - BH-series (Bharat) format: ``22BH1234AA`` (2-digit year, BH, 4 digits,
            1-2 letter vehicle class).

        Falls back to a length gate (4-10 chars) for plates that do not match
        either pattern but are clearly OCR artefacts of a real plate — this keeps
        recall high while still rejecting very short/long noise strings.
        """
        if not normalized_text:
            return False

        if _STANDARD_PLATE.match(normalized_text):
            return True
        if _BH_SERIES_PLATE.match(normalized_text):
            return True

        # Fallback: accept if length is plausible (catches obscured/partial reads)
        return 4 <= len(normalized_text) <= 10
