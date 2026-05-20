'''
Misc internal utilities

'''
import os

from datetime import datetime, timezone
from pathlib import Path
import re

import requests

from hotbaud.experimental.flock import Lock
from requests.structures import CaseInsensitiveDict


class NativeDBWarning(Warning): ...


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


epoch = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)


def path_size(path: str | Path) -> int:
    '''
    Return the byte size at the target path, if its a directory it will return
    the sum of all files under all sub-directories.

    '''
    path = Path(path)
    if path.is_file():
        return path.stat().st_size

    return sum(
        (
            subpath.stat().st_size
            for subpath in path.rglob('*')
            if subpath.is_file()
        )
    )


remote_src_protos: tuple[str, ...] = (
    'http',
    'https',
    # TODO:
    # 'ssh',
    # 'git',
    # 'git+ssh'
)


default_datadir: Path = Path.home() / '.nativedb'


def get_root_datadir() -> Path:
    return Path(os.getenv('NATIVE_DB_DATADIR', default_datadir))


def solve_redirects(
    url: str
) -> str:
    head = requests.head(url)

    # maybe follow location header (redirect)
    if redirect_url := head.headers.get('Location'):
        return redirect_url

    return url


def _sanitize_cache_token(value: str) -> str:
    value = value.strip().strip('"')
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return value


def _remote_cache_key(headers: CaseInsensitiveDict) -> str:
    etag = headers.get('ETag')
    if etag:
        if etag.startswith('W/'):
            etag = etag[2:]
        return f'etag-{_sanitize_cache_token(etag)}'

    last_modified = headers.get('Last-Modified')
    if last_modified:
        return f'mtime-{_sanitize_cache_token(last_modified)}'

    content_length = headers.get('Content-Length')
    if content_length:
        return f'len-{content_length}'

    raise RuntimeError(f'Could not derive cache key from headers: {headers}')


def fetch_remote_file(
    datadir: Path,
    url: str,
    *,
    prefix: str | None,
    suffix: str | None,
) -> Path:
    datadir.mkdir(exist_ok=True, parents=True)

    with Lock(datadir / '.download_lock'):
        head = requests.head(url, allow_redirects=True)
        head.raise_for_status()

        cache_key = _remote_cache_key(head.headers)

        if not suffix:
            url_no_params = url.split('?', 1)[0]
            url_filename = url_no_params.rsplit('/', 1)[-1]
            if '.' in url_filename:
                suffix = url_filename.rsplit('.', 1)[-1]
            else:
                suffix = 'bin'

        fname = f'{cache_key}.{suffix}'
        if prefix:
            fname = '-'.join((prefix, fname))

        local_path = datadir / fname

        if not local_path.is_file():
            local_path.parent.mkdir(parents=True, exist_ok=True)

            resp = requests.get(url, allow_redirects=True, stream=True)
            resp.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=4 * 1024):
                    if chunk:
                        f.write(chunk)

        return local_path
