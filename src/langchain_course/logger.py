import logging

from rich.logging import RichHandler


def setup_logger(name: str = "langchain_course") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
        ],
        force=True,
    )

    return logging.getLogger(name)
