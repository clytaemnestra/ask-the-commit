"""Podcast RAG: interfaces, adapters and shared plumbing.

Layering (dependencies point downwards only):

    ingest.py / rag.py / main.py / eval.py     entry points
        -> app.factory                          composition root
            -> app.providers.*                  concrete adapters
                -> app.interfaces               protocols
                    -> app.models               domain types
"""

__version__ = "0.1.0"
