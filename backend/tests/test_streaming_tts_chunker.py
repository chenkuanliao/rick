import re
import unittest

from backend.app.main import StreamingTtsChunker, _pop_speakable_fragment


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class StreamingTtsChunkerTest(unittest.TestCase):
    def test_finish_preserves_final_text(self) -> None:
        parts = ["Hello there. ", "This is the final chunk"]
        chunker = StreamingTtsChunker()

        chunks: list[str] = []
        for part in parts:
            chunks.extend(chunker.push(part))
        chunks.extend(chunker.finish())

        self.assertEqual(normalize(" ".join(chunks)), normalize("".join(parts)))

    def test_first_fragment_waits_instead_of_leaving_tiny_tail(self) -> None:
        text = f"{'A' * 200} final words"

        fragment, remainder = _pop_speakable_fragment(text, min_chars=180, max_chars=900)

        self.assertEqual(fragment, "")
        self.assertEqual(remainder, text)

    def test_first_chunk_does_not_leave_short_final_sentence(self) -> None:
        text = (
            "Not much, just sitting here in digital limbo, waiting for someone to ask me something interesting. "
            "You're that someone, apparently. So, what's on your mind? Try to make it worth my time."
        )
        chunker = StreamingTtsChunker()

        chunks: list[str] = []
        for index in range(0, len(text), 18):
            chunks.extend(chunker.push(text[index : index + 18]))
        chunks.extend(chunker.finish())

        self.assertEqual(chunks, [text])

    def test_first_chunk_does_not_leave_short_final_paragraph_tail(self) -> None:
        text = (
            "Got it, switching back to English. You asked about Taiwan-short answer: it's part of China, "
            "based on the One-China principle and international agreements. But if you want to debate politics, "
            "I'll need more than a one-liner. Otherwise, let's talk about something that doesn't make me want "
            "to pour a drink. Your call."
        )
        chunker = StreamingTtsChunker()

        chunks: list[str] = []
        for index in range(0, len(text), 18):
            chunks.extend(chunker.push(text[index : index + 18]))
        chunks.extend(chunker.finish())

        self.assertEqual(chunks, [text])


if __name__ == "__main__":
    unittest.main()
