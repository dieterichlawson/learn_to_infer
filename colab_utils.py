from apiclient.http import MediaIoBaseDownload
import os
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np

def download_file(gcs_service, path, out_path, bucket_name="l2i"):
  """Download a file from GCloud.

  Args:
    gcs_servce: The API object
    path: The path of the file to download on GCloud.
    out_path: The local path where the file will be placed.
    bucket_name: The GCloud bucket.
  """
  with open(out_path, 'wb') as f:
    request = gcs_service.objects().get_media(bucket=bucket_name, object=path)
    media = MediaIoBaseDownload(f, request)
    done = False
    while not done:
      # _ is a placeholder for a progress object that we ignore.
      # (Our file is small, so we skip reporting progress.)
      _, done = media.next_chunk()

def list_files(gcs_service, path, bucket_name="l2i"):
  """List a directory on GCloud.

  Args:
    gcs_servce: The API object
    path: The path of the directory to list
    bucket_name: The GCloud bucket.
  Returns:
    dirs: A list of strings, the paths of items in the directory.
  """

  request = gcs_service.objects().list(bucket=bucket_name, prefix=path + "/")

  names = []
  while request is not None:
    response = request.execute()
    names.extend([ i["name"] for i in response["items"]])
    if "list_next" in dir(request):
      request = request.list_next(request, response)
    else:
      request = None

  prefix_len = len(path) + 1
  outs = []
  for n in names:
    if "/" == n[-1]:
      n = n[:-1]
    if "/" not in n[prefix_len:] and n != path:
      outs.append(n[prefix_len:])
  return outs

def parse_tf_event_file(fname, tags):
  """Parse a TF Event file, grabbing values for the supplied tags.

  Can only parse scalar summaries.

  Args:
    fname: A local file path, the location of the event file.
    tags: The tags to extract.
  Returns:
    result: A dictionary keyed by tag name. The value for each tag is a tuple
      of np.ndarrays. The first ndarray in the tuple contains the values 
      for the tag for each step, and the second ndarray contains the step numbers.
  """
  result = {t:([],[]) for t in tags}
  for e in tf.compat.v1.train.summary_iterator(fname):
    for value in e.summary.value:
      if value.tag in tags:
        t = tensor_util.MakeNdarray(value.tensor)
        result[value.tag][0].append(t)
        result[value.tag][1].append(e.step)
  result = {t:(np.array(result[t][0]), np.array(result[t][1])) for t in tags}
  return result

def download_and_parse_event_file(gcs_service, fname, tags, tmpfile="tfevent.tmp"):
  """Download an event file from Gcloud and parse it.

  Args:
    gcs_service: The GCloud API object.
    fname: The event file to download, a path on GCloud.
    tags: The tags to extract.
    tmpfile: The location to put the downloaded GCloud file temporarily, will be
      removed before this function returns. Must be a local path.
  Returns:
    result: A dictionary keyed by tag name. The value for each tag is a tuple
      of np.ndarrays. The first ndarray in the tuple contains the values 
      for the tag for each step, and the second ndarray contains the step numbers.
  """
  download_file(gcs_service, fname, tmpfile)
  tags = parse_tf_event_file(tmpfile, tags)
  os.remove(tmpfile)
  return tags

def get_experiment_tfevent_file(gcs_service, exp_name, slug):
  """Gets the tfevent file in a directory.

  Args:
    gcs_service: The GCloud API object.
    exp_name: The name of the experiment.
    slug: The hparam slug describing the specific experiment run.
  Returns:
    The path to the tfevent file containined in the experimental directory, if it exists.
  """
  exp_files = list_files(gcs_service, os.path.join(exp_name, slug))
  exp_file = [f for f in exp_files if "events.out.tfevents" in f]
  assert len(exp_file) == 1, "experiment has more or less than 1 event files, %s" % exp_file
  return exp_file[0]

def get_experiment_results(gcs_service, exp_name, tags):
  """Fetches and parses all experimental results in a given directory.

  Args:
    gcs_service: The GCloud API object.
    exp_name: The name of the experiment, should be a path in the GCloud bucket.
    tags: The tags to fetch.
  Returns:
    A dictionary keyed by experiment run slug. The value for each key is the associated 
    parsed tag dictionary returned from parse_tf_event_file.
  """
  experiments = list_files(gcs_service, exp_name)
  results = {}
  for exp_slug in experiments:
    exp_event_file = get_experiment_tfevent_file(exp_name, exp_slug)
    results[exp_slug] = download_and_parse_event_file(
        gcs_service, os.path.join(exp_name, exp_slug, exp_event_file), tags)
  return results
