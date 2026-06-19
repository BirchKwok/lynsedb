class RustResponse:
    def __init__(self, status_code, content, json_ok=False, json_value=None):
        self.status_code = int(status_code)
        self.content = bytes(content)
        self._json_ok = bool(json_ok)
        self._json_value = json_value
        self.headers = {
            "Content-Type": "application/json" if self._json_ok else "application/octet-stream"
        }

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        if not self._json_ok:
            raise ValueError("response body is not JSON")
        return self._json_value


class RustRemoteSession:
    def __init__(self, base_url="", api_key=None):
        from .. import _core as _lynse_core

        self._client = _lynse_core.RemoteHttpClient(base_url or "", api_key)

    @staticmethod
    def _content_type(headers, default=None):
        if not headers:
            return default
        for key, value in headers.items():
            if str(key).lower() == "content-type":
                return str(value)
        return default

    @staticmethod
    def _wrap(raw):
        status_code, content, json_ok, json_value = raw
        return RustResponse(status_code, content, json_ok, json_value)

    def post(self, url, data=None, headers=None, content=None, json=None, params=None):
        if data is not None:
            raise NotImplementedError("RustRemoteSession does not support form-encoded data")
        if content is not None:
            content_type = self._content_type(headers, "application/octet-stream")
            return self._wrap(self._client.post_binary_raw(url, content, params, content_type))
        return self._wrap(self._client.post_json(url, json if json is not None else {}, params))

    def get(self, url, headers=None, params=None):
        return self._wrap(self._client.get(url, params))

    def request(
        self,
        method,
        url,
        *,
        json=None,
        params=None,
        content=None,
        headers=None,
    ):
        method = str(method).upper()
        if method == "GET":
            return self.get(url, headers=headers, params=params)
        if method == "POST":
            return self.post(url, headers=headers, content=content, json=json, params=params)
        raise NotImplementedError(f"RustRemoteSession only supports GET/POST, got {method}")

    def close(self):
        self._client.close()


class Poster:
    def __init__(self, retries=3, timeout=None, http2=False, api_key=None):
        """
        Thin compatibility wrapper for the Rust remote HTTP client.

        retries/timeout/http2 are accepted for API compatibility; connection
        management is handled by the Rust-side persistent client.
        """
        self.session = RustRemoteSession(api_key=api_key)

    def post(self, url, data=None, headers=None, content=None, json=None, params=None):
        return self.session.post(url, data=data, headers=headers, content=content, json=json, params=params)

    def get(self, url, headers=None, params=None):
        return self.session.get(url, headers=headers, params=params)

    def close(self):
        self.session.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
