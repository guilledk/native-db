'''
Misc internal utilities

'''
from datetime import datetime, timezone
import os
from pathlib import Path

import requests


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


def fetch_remote_file(
    datadir: Path, url: str, *, prefix: str | None, suffix: str | None
) -> tuple[str, Path]:
    # perform head requests looking for ETag header with checksum
    head = requests.head(url)

    # maybe follow location header (redirect)
    if redirect_url := head.headers.get('Location'):
        head = requests.head(redirect_url)
        url = redirect_url

    # expect etag checksum
    etag = head.headers.get('ETag')
    if not etag:
        raise RuntimeError(
            f'Remote source head response missing etag header: {head.headers}'
        )

    # maybe we got a "weak etag" which is prefixed by 'W/'
    if etag.startswith('W/'):
        etag = etag[2:]

    # strip quotes
    etag = etag.strip('"')

    # maybe figure out prefix and suffix from url
    if not prefix or not suffix:
        url_no_params = url.split('?')[0]
        url_filename = url_no_params.split('/')[-1]
        filename_parts = url_filename.split('.')

        pre, suf = filename_parts

        if not prefix:
            prefix = pre

        if not suffix:
            suffix = suf

    # finally generate etag based local cache path
    local_path = datadir / etag / f'{prefix}.{suffix}'

    # if local file missing, attempt download
    if not local_path.is_file():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        resp = requests.get(url, allow_redirects=True, stream=True)
        resp.raise_for_status()

        with open(local_path, 'wb+') as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024):
                f.write(chunk)

    return url, local_path


class NativeDBWarning(Warning):
    ...
