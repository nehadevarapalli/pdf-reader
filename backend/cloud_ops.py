import os

import boto3

# Initialize the S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SECRET_KEY')
)


def download_file_from_s3(object_key, local_dir, bucket_name='neu-pdf-webpage-parser'):
    '''
    Download a file from S3 and save it locally in a directory that mirrors the S3 path.
    '''
    local_path = os.path.join(local_dir, object_key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket_name, object_key, local_path)
        print(f"Downloaded '{object_key}' from '{bucket_name}' to '{local_path}'.")
        return local_path
    except Exception as e:
        print(f"Error downloading file '{object_key}': {e}")


def upload_file_to_s3(file_path, object_key, bucket_name='neu-pdf-webpage-parser'):
    '''
    Upload a single file to S3.
    '''
    try:
        s3.upload_file(file_path, bucket_name, object_key)
        print(f"Uploaded '{file_path}' to '{bucket_name}/{object_key}'.")
    except Exception as e:
        print(f"Error uploading file '{file_path}': {e}")


def upload_directory_to_s3(directory_path, prefix, bucket_name='neu-pdf-webpage-parser'):
    '''
    Upload all files in a directory to S3 with a specified prefix.
    '''
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            object_key = os.path.join(prefix, os.path.relpath(file_path, directory_path))
            upload_file_to_s3(file_path, object_key, bucket_name)


def add_tags_to_object(object_key, tags, bucket_name='neu-pdf-webpage-parser'):
    '''
    Add metadata tags to an S3 object.
    '''
    try:
        s3.put_object_tagging(
            Bucket=bucket_name,
            Key=object_key,
            Tagging={
                'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]
            }
        )
        print(f"Tags added to '{bucket_name}/{object_key}'.")
    except Exception as e:
        print(f"Error adding tags to '{object_key}': {e}")


def upload_file_with_encyption(file_path, object_key, bucket_name='neu-pdf-webpage-parser'):
    '''
    Upload a file to S3 with server-side encryption.
    '''
    s3 = boto3.client('s3')
    try:
        s3.upload_file(
            file_path,
            bucket_name,
            object_key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256'
            }
        )
        print(f"File '{file_path}' has been uploaded to '{bucket_name}/{object_key}' with encryption.")
    except Exception as e:
        print(f"Error uploading file with encryption: {e}")
