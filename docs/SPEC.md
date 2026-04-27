# MediaRef Specification 1.0

**Status:** Stable. Schema is permanent under [semver](https://semver.org/) major version 1.
**Scope:** Wire-format and semantics for a single media reference. Does not
specify decoding implementations, transport, or container formats.

A *MediaRef* is a 2-tuple `(uri, pts_ns)` that points to either a still image
or a single frame of a video. Implementations MAY add transient state (cached
buffers, resolved paths) but MUST NOT add fields to the wire format.

---

## 1. Schema

| Field    | Type                  | Required | Description                                                              |
| -------- | --------------------- | -------- | ------------------------------------------------------------------------ |
| `uri`    | string                | yes      | Locator for the media resource. Grammar in §2.                           |
| `pts_ns` | int64 \| null         | no       | Frame presentation timestamp in nanoseconds. `null`/absent ⇒ still image. |

No other fields are part of the wire format. Producers MUST NOT emit
additional fields; consumers SHOULD reject unknown fields or strip them.

### 1.1 Arrow / Parquet representation

```
struct<uri: string, pts_ns: int64>   -- pts_ns nullable
```

### 1.2 JSON representation

```json
{"uri": "video.mp4", "pts_ns": 1500000000}
{"uri": "image.png"}                          // pts_ns absent ⇒ still image
{"uri": "image.png", "pts_ns": null}          // equivalent to the above
```

`pts_ns` MAY be omitted when `null`. Producers SHOULD prefer omission for size.

---

## 2. URI grammar

`uri` MUST be one of the following forms.

### 2.1 Absolute URI per RFC 3986

[RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986) absolute-URI with
any scheme. Schemes a compliant MediaRef library is expected to recognize:

| Scheme           | Meaning                                                       |
| ---------------- | ------------------------------------------------------------- |
| `file://`        | Local file URI.                                               |
| `http://`        | Remote resource over HTTP.                                    |
| `https://`       | Remote resource over HTTPS.                                   |
| `data:`          | Embedded media per RFC 2397 (see §2.2).                       |

Implementations SHOULD delegate any URI whose scheme is not in the
table above to an extensible URI handler such as
[fsspec](https://filesystem-spec.readthedocs.io) — i.e. open-set, so
schemes like `s3://`, `gs://`, `gcs://`, `hf://`, `az://`, `azure://`,
`abfs(s)://`, `adl://`, `r2://`, `ftp://`, `sftp://`, `ssh://`,
`memory://`, `webdav://`, `gdrive://`, `ipfs://` (and any future fsspec
backend) all work without scheme-specific code. The reference
implementation requires fsspec as a core dependency; each backend
(`s3fs`, `gcsfs`, `huggingface_hub`, `adlfs`, …) must be installed
separately for the schemes it serves.

### 2.2 Data URI per RFC 2397

[RFC 2397](https://datatracker.ietf.org/doc/html/rfc2397) form:

```
data:[<mediatype>][;base64],<data>
```

The `<mediatype>` SHOULD be `image/*` for embedded images. Embedded video is
permitted but discouraged due to size; for video, prefer a `file://` or
remote-URL reference plus `pts_ns`.

### 2.3 POSIX path (relative or absolute)

A `uri` that is not an absolute URI per RFC 3986 is interpreted as a POSIX
filesystem path. Forward-slash separator (`/`). Both relative
(`videos/clip.mp4`) and absolute (`/data/clip.mp4`) paths are valid.

Resolution of relative paths is the consumer's responsibility and depends on
deployment context. Implementations SHOULD provide a `resolve_relative_path`
helper that takes a base directory.

---

## 3. `pts_ns` semantics

`pts_ns` is the **presentation timestamp** of the requested frame in
nanoseconds, measured from the **start of the stream as reported by the
container** (`begin_stream_seconds`). It is NOT measured from the wall clock,
NOT from epoch, NOT from any episode boundary.

### 3.1 Type and range

- Type: signed 64-bit integer (`int64`).
- Required range: implementations MUST handle any value in
  `[0, 2^63 − 1]`, which covers any practical video duration.
- Negative values are reserved and SHOULD NOT be produced.

### 3.2 Frame selection rule

Given a video with frames whose presentation timestamps are
`pts[0] < pts[1] < … < pts[N-1]`, a query at `pts_ns = t`:

```
pts[i] ≤ t < pts[i+1]   ⇒   returns frame i
pts[N-1] ≤ t < end_stream_ns   ⇒   returns frame N-1 (last frame)
t < pts[0]   or   t ≥ end_stream_ns   ⇒   error
```

This is the [TorchCodec](https://github.com/pytorch/torchcodec) "what is on
screen at time t" model. Frames are displayed starting at their PTS until the
next frame's PTS replaces them; the duration field is ignored. See
[`playback_semantics.md`](playback_semantics.md) for worked examples and
edge cases.

### 3.3 Still image vs. video frame

- `pts_ns is None` (or absent) ⇒ the resource is interpreted as a **still
  image**. The MIME type is determined by the `uri` (extension or
  `data:` mediatype).
- `pts_ns is not None` ⇒ the resource is interpreted as **video** and the
  frame at `pts_ns` is the referent.

A producer MUST NOT set `pts_ns` for a resource that is not a video stream.

---

## 4. Equivalence

Two MediaRefs are **equivalent** iff:

1. Their `uri` strings are byte-equal **after** RFC 3986 normalization
   (case-fold scheme/host, percent-decode unreserved characters, remove dot
   segments). For `data:` URIs the entire URI is compared verbatim.
2. Their `pts_ns` values are equal as integers, treating `null` and absent
   as the same.

Two equivalent MediaRefs MUST resolve to the same media. Implementations MAY
short-circuit decoding for equivalent refs.

---

## 5. Serialization rules

A compliant serializer MUST produce one of the representations in §1.1 or
§1.2. Specifically:

- The JSON form MUST use UTF-8 and standard JSON number for `pts_ns`.
- Producers SHOULD omit `pts_ns` when null. Consumers MUST accept both
  `pts_ns: null` and an absent `pts_ns` field.
- Field ordering is not significant.
- The Arrow form MUST use the exact field names `uri` and `pts_ns` and the
  exact types in §1.1.

---

## 6. "MediaRef-compliant" libraries and datasets

A library is **MediaRef-compliant 1.0** iff:

1. It can read and write the JSON form in §1.2.
2. Its semantics match §3.
3. It does not extend the wire schema with additional required fields.

A dataset is **MediaRef-compliant 1.0** iff every column declared as a media
reference contains values matching §1.1 or §1.2.

Compliant libraries SHOULD declare compliance in their README and tag their
HuggingFace Hub datasets with `mediaref`.

---

## 7. Versioning

The schema in §1 is frozen for the lifetime of MediaRef Spec 1.x. Any
breaking change (renaming a field, changing types, removing the optional
nature of `pts_ns`) requires a new major version (2.0).

Backward-compatible additions (new optional fields, new recognized URI
schemes in §2.1) are permitted in 1.x minor revisions.

---

## 8. Reference implementation

The reference implementation is the
[`mediaref`](https://github.com/open-world-agents/MediaRef) Python package.
Other implementations may exist; conformance is determined solely by this
specification, not by behavioral compatibility with any single
implementation.

---

## Citation

Cite this specification as:

```bibtex
@software{mediaref,
  author = {Choi, Suhwan},
  title  = {MediaRef: a portable frame-level media reference primitive},
  url    = {https://github.com/open-world-agents/MediaRef},
  year   = {2025}
}
```
