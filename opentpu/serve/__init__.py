"""otpu-serve: an OpenAI-style API server for one model on openTPU (docs/serve.md).

kvio      a context's bytes: the KV caches' pages and the state they do not hold, read from
          and written to the device
store     the KV store on the host: pages in RAM, spilled to files on disk, and the prefix
          index that finds the longest cached prefix of a prompt
scheduler the device's worker: requests admitted, their contexts switched in and out of the
          device, prefill and decode interleaved between them
api       the HTTP server: /v1/chat/completions, /v1/completions, /v1/models
tools     OpenAI tool calls over the template's format (qwen3_coder, LFM2's): the request's
          tools and tool messages to the template, a reply's calls parsed (also as it streams)
"""
