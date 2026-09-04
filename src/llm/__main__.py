"""Ask the configured backend one question, to prove it is reachable."""

from __future__ import annotations

import argparse
import logging

from llm import PROVIDERS, load_provider

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Reply with the single word: reachable.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provider = load_provider(args.provider, model=args.model, base_url=args.base_url)
    logger.info("%s / %s", provider.name, provider.config.model)

    answer = provider.complete(args.prompt)
    logger.info("%s", answer.text)
    logger.info("%d in, %d out", answer.input_tokens, answer.output_tokens)


if __name__ == "__main__":
    main()
