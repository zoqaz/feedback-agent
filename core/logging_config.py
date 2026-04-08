import logging

class ColorFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG:    "\x1b[38;5;244m",
        logging.INFO:     "\x1b[38;5;39m",
        logging.WARNING:  "\x1b[38;5;214m",
        logging.ERROR:    "\x1b[38;5;196m",
        logging.CRITICAL: "\x1b[1;38;5;196m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        levelname = record.levelname
        record.levelname = f"{color}{levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = levelname


def configure_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return  # avoid double configuration

    handler = logging.StreamHandler()
    handler.setFormatter(
        ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(handler)