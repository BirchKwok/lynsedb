import httpx


class Poster:
    def __init__(self, retries=3, timeout=None, http2=False):
        """
        A class for making HTTP requests.

        Parameters:
            retries (int): The number of times to retry the request.
            timeout (int or None): The request timeout in seconds.
            http2 (bool): Whether to use HTTP/2.
        """
        self.retries = retries
        self.timeout = timeout
        self.http2 = http2
        transport = httpx.HTTPTransport(retries=retries, http2=http2)
        self.session = httpx.Client(transport=transport, timeout=timeout)

    def post(self, url, data=None, headers=None, content=None, json=None):
        """
        Make a POST request.

        Parameters:
            url (str): The URL to request.
            data (dict or None): The data to send.
            headers (dict or None): The headers to send.
            content (str or None or btypes): The content to send.
            json (dict or None): The JSON data to send.

        Returns:
            httpx.Response: The response object.
        """
        return self.session.post(url, data=data, headers=headers, content=content, json=json)

    def get(self, url, headers=None):
        """
        Make a GET request.

        Parameters:
            url (str): The URL to request.
            headers (dict or None): The headers to send.

        Returns:
            httpx.Response: The response object.
        """
        return self.session.get(url, headers=headers)
