"""s3 utility functions."""
from __future__ import annotations

import _pickle as cPickle
import io
import os
from collections import namedtuple
from operator import attrgetter

import pandas as pd
import polars as pl
import pyarrow.dataset as ds

S3Obj = namedtuple("S3Obj", ["key", "mtime", "size", "ETag"])


def save_pandas_to_s3_parquet(s3, bucket, f, df):
    """Save pandas dataframe in parquet format to s3.

    Args:
        s3 (boto3.resource)
            usually initialised by:
                s3 = boto3.resource(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    )
        bucket (str)
            bucket for data
        df (pd.DataFrame)
            pandas dataframe only
        f (str)
            file_url (excluding the bucket)

    Returns:
        None

    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    return s3.Object(bucket, f).put(Body=buffer.getvalue())


def save_polars_to_s3_parquet(fs, bucket, f, df):
    """Save polars dataframe in parquet format to s3.

    Args:
        fs (s3fs.S3FileSystem)
            ususally initialised by:
                fs = s3fs.S3FileSystem(
                    key=AWS_ACCESS_KEY_ID,
                    secret=AWS_SECRET_ACCESS_KEY,
                    use_ssl=True
                )
        bucket (str)
            s3 bucket for storing data
        f (str)
            file url, excluding the bucket name
        df (pl.DataFrame)
            polars DataFrame

    Returns:
        None

    """
    # write parquet
    with fs.open(f"{bucket}/{f}", mode="wb") as f:
        df.write_parquet(f)


def read_s3_parquet_to_pandas(s3, bucket, f):
    """Load parquet from s3 to pandas dataframe.

    Args:
        s3 (boto3.resource)
            usually initialised by:
                s3 = boto3.resource(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    )
        bucket (str)
            bucket for data
        f (str)
            file_url (excluding the bucket)

    Returns:
        pandas.DataFrame
    """
    buffer = io.BytesIO()
    obj = s3.Object(bucket, f)
    obj.download_fileobj(buffer)
    return pd.read_parquet(buffer)


def read_s3_parquet_to_polars(fs, bucket, f):
    """Load parquet from s3 to polars dataframe.

    Args:
        fs (s3fs.S3FileSystem)
            ususally initialised by:
                fs = s3fs.S3FileSystem(
                    key=AWS_ACCESS_KEY_ID,
                    secret=AWS_SECRET_ACCESS_KEY,
                    use_ssl=True
                )
        bucket (str)
            s3 bucket for storing data
        f (str)
            file url, excluding the bucket name

    Returns:
        polars.DataFrame

    """
    dataset = ds.dataset(f"s3://{bucket}/{f}", filesystem=fs, format="parquet")
    return pl.scan_pyarrow_dataset(dataset)


def save_s3_cpickle(s3, bucket, f, data):
    """Save object in cPickle format in s3.

    Args:
        s3 (boto3.resource)
            usually initialised by:
                s3 = boto3.resource(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    )
        bucket (str)
            s3 bucket for storing data
        f (str)
            file url, excluding the bucket name
        data (Any)
            any picklable data type / objects that can be serialised

    Returns:
        None

    """
    pickle_byte_obj = cPickle.dumps(data)
    return s3.Object(bucket, f).put(Body=pickle_byte_obj)


def read_s3_cpickle(s3, bucket, f):
    """Load cPickle object from s3.

    Args:
        s3 (boto3.resource)
            usually initialised by:
                s3 = boto3.resource(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    )
        bucket (str)
            s3 bucket for storing data
        f (str)
            file url, excluding the bucket name

    Returns:
        Any
            any picklable data type / objects that can be serialised

    """
    return cPickle.loads(s3.Bucket(bucket).Object(f).get()["Body"].read())


def s3_list(
    s3,
    bucket,
    path,
    start=None,
    end=None,
    recursive=True,
    list_dirs=True,
    list_objs=True,
    limit=None,
):
    """List files in s3 location.

    Iterator that lists a bucket's objects under path, (optionally) starting with
    start and ending before end.

    If recursive is False, then list only the "depth=0" items (dirs and objects).

    If recursive is True, then list recursively all objects (no dirs).

    Sauce: https://stackoverflow.com/questions/35803027/
    retrieving-subfolders-names-in-s3-bucket-from-boto3

    Args:
        s3:
            s3 resource - boto3.resource("s3")
        bucket:
            string of bucket name.
        path:
            a directory in the bucket.
        start:
            optional: start key, inclusive (may be a relative path under path, or
            absolute in the bucket)
        end:
            optional: stop key, exclusive (may be a relative path under path, or
            absolute in the bucket)
        recursive:
            optional, default True. If True, lists only objects. If False, lists
            only depth 0 "directories" and objects.
        list_dirs:
            optional, default True. Has no effect in recursive listing. On
            non-recursive listing, if False, then directories are omitted.
        list_objs:
            optional, default True. If False, then directories are omitted.
        limit:
            optional. If specified, then lists at most this many items.

    Returns:
        an iterator of S3Obj.

    Examples:
        # set up
        >>> bucket = 'bar'

        # iterate through all S3 objects under some dir
        >>> for p in s3ls(bucket, 'some/dir'):
        ...     print(p)

        # iterate through up to 20 S3 objects under some dir, starting with foo_0010
        >>> for p in s3ls(bucket, 'some/dir', limit=20, start='foo_0010'):
        ...     print(p)

        # non-recursive listing under some dir:
        >>> for p in s3ls(bucket, 'some/dir', recursive=False):
        ...     print(p)

        # non-recursive listing under some dir, listing only dirs:
        >>> for p in s3ls(bucket, 'some/dir', recursive=False, list_objs=False):
        ...     print(p)

    """
    bucket = s3.Bucket(bucket)
    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
        # note: need to use a string just smaller than start, because
        # the list_object API specifies that start is excluded (the first
        # result is *after* start).
        kwargs.update(Marker=__prev_str(start))
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter="/")
        if not path.endswith("/"):
            path += "/"
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={"MaxItems": limit})

    paginator = bucket.meta.client.get_paginator("list_objects")
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if "CommonPrefixes" in resp and list_dirs:
            q = [S3Obj(f["Prefix"], None, None, None) for f in resp["CommonPrefixes"]]
        if "Contents" in resp and list_objs:
            q += [
                S3Obj(f["Key"], f["LastModified"], f["Size"], f["ETag"])
                for f in resp["Contents"]
            ]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter("key"))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            yield p


def __prev_str(s):
    """Helper function for s3list.

    Sauce: https://stackoverflow.com/questions/35803027/
    retrieving-subfolders-names-in-s3-bucket-from-boto3

    """
    if len(s) == 0:
        return s
    s, c = s[:-1], ord(s[-1])
    if c > 0:
        s += chr(c - 1)
    s += "".join(["\u7fff" for _ in range(10)])
    return s


def get_latest_s3obj(s3, bucket, prefix, obj_type, full_path=True):
    """Finds latest s3 object within a folder.

    For some s3 prefix, finds the latest s3 object or folder in
    that location. Latest is sorted by time

    Args:
        bucket (str):
            name of s3 bucket
        prefix (str):
            folder / prefix where we want to search the files
        obj_type (str):
            can either be "file" OR "folder" to indicate what to return
        full_path (bool):
            can either be True OR False to show whether we want to return
            the full file path (inluding s3 location) or just the prefix
            (i.e. excluding s3 location)

    Returns:
        str:
            the full or partial string path

    """
    if obj_type == "file":
        files = [x[0] for x in list(s3_list(s3, bucket, prefix, list_dirs=False))]
        f = files[-1]
        if full_path:
            return f"s3://{bucket}/{f}"
        else:
            return f
    elif obj_type == "folder":
        files = [
            x[0]
            for x in list(
                s3_list(
                    s3,
                    bucket,
                    prefix,
                    list_dirs=True,
                    recursive=False,
                    list_objs=False,
                ),
            )
        ]
        return files[-1]
