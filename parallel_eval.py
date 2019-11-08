# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

from absl import logging
import eval_util
import frame_level_models
import nextvlad
import losses
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import device_lib
import utils
import video_level_models
import numpy as np

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Dataset flags.
    flags.DEFINE_string(
        "train_dir", "/tmp/yt8m_model/",
        "The directory to load the model files from. "
        "The tensorboard metrics files are also saved to this "
        "directory.")
    flags.DEFINE_string(
        "eval_data_pattern", "",
        "File glob defining the evaluation dataset in tensorflow.SequenceExample "
        "format. The SequenceExamples are expected to have an 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_bool(
        "segment_labels", False,
        "If set, then --eval_data_pattern must be frame-level features (but with"
        " segment_labels). Otherwise, --eval_data_pattern must be aggregated "
        "video-level features. The model must also be set appropriately (i.e. to "
        "read 3D batches VS 4D batches.")

    flags.DEFINE_integer(
        "num_gpu", 1, "The maximum number of GPU devices to use for training. "
    )

    flags.DEFINE_float("final_temperature", 1.0, "the temperature of final online learning")
    flags.DEFINE_float("final_lambda", 1.0, "the lambda of final online learning")

    # Other flags.
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch.")
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")
    flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
    """Creates the section of the graph which reads the evaluation data.

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_readers: How many I/O threads to use.

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of %d for evaluation.", batch_size)
    with tf.name_scope("eval_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        logging.info("number of evaluation files: %d", len(files))
        filename_queue = tf.train.string_input_producer(
            files, shuffle=False, num_epochs=1)
        eval_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
        ]
        return tf.train.batch_join(
            eval_data,
            batch_size=batch_size,
            capacity=3 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def build_graph(reader,
                model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
    """Creates the Tensorflow graph for evaluation.

    Args:
      reader: The data file reader. It should inherit from BaseReader.
      model: The core model (e.g. logistic or neural net). It should inherit from
        BaseModel.
      eval_data_pattern: glob path to the evaluation data files.
      label_loss_fn: What kind of loss to apply to the model. It should inherit
        from BaseLoss.
      batch_size: How many examples to process at a time.
      num_readers: How many threads to use for I/O operations.
    """

    global_step = tf.Variable(0, trainable=False, name="global_step")
    input_data_dict = get_input_evaluation_tensors(
        reader, eval_data_pattern, batch_size=batch_size, num_readers=num_readers)
    video_id_batch = input_data_dict["video_ids"]
    model_input_raw = input_data_dict["video_matrix"]
    labels_batch = input_data_dict["labels"]
    num_frames = input_data_dict["num_frames"]
    tf.summary.histogram("model_input_raw", model_input_raw)

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = "/gpu:%d"
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = "/cpu:%d"
    print("flags!!!", device_string)
    # feature_dim = len(model_input_raw.get_shape()) - 1

    # Normalize input features.
    # model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    if FLAGS.segment_labels:
        label_weights = input_data_dict["label_weights"]
    else:
        label_weights = None

    offset = np.array([4. / 512] * 1024 + [0] * 128)
    offset = tf.constant(offset, dtype=tf.float32)

    eigen_val = tf.constant(np.sqrt(np.load("yt8m_pca/eigenvals.npy")[:1024, 0]), dtype=tf.float32)
    model_input = tf.multiply(model_input_raw - offset, tf.pad(eigen_val + 1e-4, [[0, 128]], constant_values=1.))

    tower_logits = []

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.variable_scope("tower_%d" % i, reuse=False):
                result = model.create_model(
                    model_input,
                    num_frames=num_frames,
                    vocab_size=reader.num_classes,
                    labels=labels_batch,
                    is_training=False)
                logits = result["logits"]
                tower_logits.append(logits)

    with tf.device(device_string % 0):
        with tf.variable_scope("ensemble"):
            ftr_mean = tf.reduce_mean(model_input, axis=1)
            print("ftr mean shape: ", ftr_mean.get_shape().as_list())
            ftr_mean = slim.batch_norm(ftr_mean,
                                       center=True,
                                       scale=True,
                                       fused=False,
                                       is_training=False,
                                       scope="mix_weights_bn")
            mix_weights = slim.fully_connected(ftr_mean, num_towers, activation_fn=None,
                                               weights_initializer=slim.variance_scaling_initializer(),
                                               scope="mix_weights")
            mix_weights = tf.nn.softmax(mix_weights, axis=-1)
            tf.summary.histogram("mix_weights", mix_weights)

            logits = tf.stack(tower_logits, axis=1)
            final_logit = tf.reduce_sum(tf.multiply(logits, tf.expand_dims(mix_weights, axis=-1)), axis=1,
                                         keepdims=False)
            final_predictions = tf.nn.sigmoid(final_logit)

        final_label_loss = label_loss_fn.calculate_loss(final_predictions, labels_batch, label_weights=label_weights)

        tf.summary.scalar("label_loss", final_label_loss)
        tf.add_to_collection("global_step", global_step)
        tf.add_to_collection("loss", final_label_loss)
        tf.add_to_collection("predictions", final_predictions)
        tf.add_to_collection("input_batch", model_input)
        tf.add_to_collection("input_batch_raw", model_input_raw)
        tf.add_to_collection("video_id_batch", video_id_batch)
        tf.add_to_collection("num_frames", num_frames)
        tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
        if FLAGS.segment_labels:
            tf.add_to_collection("label_weights", input_data_dict["label_weights"])
        tf.add_to_collection("summary_op", tf.summary.merge_all())


def evaluation_loop(fetches, saver, summary_writer, evl_metrics,
                    last_global_step_val):
    """Run the evaluation loop once.

    Args:
      fetches: a dict of tensors to be run within Session.
      saver: a tensorflow saver to restore the model.
      summary_writer: a tensorflow summary_writer
      evl_metrics: an EvaluationMetrics object.
      last_global_step_val: the global step used in the previous evaluation.

    Returns:
      The global_step used in the latest model.
    """

    global_step_val = -1
    with tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(
                allow_growth=True),allow_soft_placement=True)) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
        if latest_checkpoint:
            logging.info("Loading checkpoint for eval: %s", latest_checkpoint)
            # Restores from checkpoint
            saver.restore(sess, latest_checkpoint)
            # Assuming model_checkpoint_path looks something like:
            # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
            global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]

            # Save model
            if FLAGS.segment_labels:
                inference_model_name = "segment_inference_model"
            else:
                inference_model_name = "inference_model"
            saver.save(
                sess,
                os.path.join(FLAGS.train_dir, "inference_model", inference_model_name))
        else:
            logging.info("No checkpoint file found.")
            return global_step_val

        if global_step_val == last_global_step_val:
            logging.info(
                "skip this checkpoint global_step_val=%s "
                "(same as the previous one).", global_step_val)
            return global_step_val

        sess.run([tf.local_variables_initializer()])

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(
                    qr.create_threads(sess, coord=coord, daemon=True, start=True))
            logging.info("enter eval_once loop global_step_val = %s. ",
                         global_step_val)

            evl_metrics.clear()

            examples_processed = 0
            while not coord.should_stop():
                batch_start_time = time.time()
                output_data_dict = sess.run(fetches)
                seconds_per_batch = time.time() - batch_start_time
                labels_val = output_data_dict["labels"]
                summary_val = output_data_dict["summary"]
                example_per_second = labels_val.shape[0] / seconds_per_batch
                examples_processed += labels_val.shape[0]

                predictions = output_data_dict["predictions"]
                if FLAGS.segment_labels:
                    # This is a workaround to ignore the unrated labels.
                    predictions *= output_data_dict["label_weights"]
                iteration_info_dict = evl_metrics.accumulate(predictions, labels_val,
                                                             output_data_dict["loss"])
                iteration_info_dict["examples_per_second"] = example_per_second

                iterinfo = utils.AddGlobalStepSummary(
                    summary_writer,
                    global_step_val,
                    iteration_info_dict,
                    summary_scope="Eval")
                logging.info("examples_processed: %d | %s", examples_processed,
                             iterinfo)

        except tf.errors.OutOfRangeError as e:
            logging.info(
                "Done with batched inference. Now calculating global performance "
                "metrics.")
            # calculate the metrics for the entire epoch
            epoch_info_dict = evl_metrics.get()
            epoch_info_dict["epoch_id"] = global_step_val

            summary_writer.add_summary(summary_val, global_step_val)
            epochinfo = utils.AddEpochSummary(
                summary_writer,
                global_step_val,
                epoch_info_dict,
                summary_scope="Eval")
            logging.info(epochinfo)
            evl_metrics.clear()
        except Exception as e:  # pylint: disable=broad-except
            logging.info("Unexpected exception: %s", str(e))
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        logging.info("Total: examples_processed: %d", examples_processed)

        return global_step_val


def evaluate():
    """Starts main evaluation loop."""
    tf.set_random_seed(0)  # for reproducibility

    # Write json of flags
    model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
    if not file_io.file_exists(model_flags_path):
        raise IOError(("Cannot find file %s. Did you run train.py on the same "
                       "--train_dir?") % model_flags_path)
    flags_dict = json.loads(file_io.FileIO(model_flags_path, mode="r").read())

    with tf.Graph().as_default():
        # convert feature_names and feature_sizes to lists of values
        feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            flags_dict["feature_names"], flags_dict["feature_sizes"])

        if flags_dict["frame_features"]:
            reader = readers.YT8MFrameFeatureReader(
                feature_names=feature_names,
                feature_sizes=feature_sizes,
                segment_labels=FLAGS.segment_labels)
        else:
            reader = readers.YT8MAggregatedFeatureReader(
                feature_names=feature_names, feature_sizes=feature_sizes)

        model = find_class_by_name(flags_dict["model"],
                                   [frame_level_models, video_level_models, nextvlad])()
        label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

        if not FLAGS.eval_data_pattern:
            raise IOError("'eval_data_pattern' was not specified. Nothing to "
                          "evaluate.")

        build_graph(
            reader=reader,
            model=model,
            eval_data_pattern=FLAGS.eval_data_pattern,
            label_loss_fn=label_loss_fn,
            num_readers=FLAGS.num_readers,
            batch_size=FLAGS.batch_size)
        logging.info("built evaluation graph")

        # A dict of tensors to be run in Session.
        fetches = {
            "video_id": tf.get_collection("video_id_batch")[0],
            "predictions": tf.get_collection("predictions")[0],
            "labels": tf.get_collection("labels")[0],
            "loss": tf.get_collection("loss")[0],
            "summary": tf.get_collection("summary_op")[0]
        }
        if FLAGS.segment_labels:
            fetches["label_weights"] = tf.get_collection("label_weights")[0]

        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.train_dir, "eval"), graph=tf.get_default_graph())

        print('num_classes: %s' %reader.num_classes)
        evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)

        last_global_step_val = -1
        while True:
            last_global_step_val = evaluation_loop(fetches, saver, summary_writer,
                                                   evl_metrics, last_global_step_val)
            if FLAGS.run_once:
                break


def main(unused_argv):
    logging.set_verbosity(tf.logging.INFO)
    logging.info("tensorflow version: %s", tf.__version__)
    evaluate()


if __name__ == "__main__":
    # import sys
    # argv = sys.argv
    # train_dir = "test_nextvlad"
    # datapath = "/media/linrongc/dream/data/yt8m/frame/2/validate"
    # options = """
    #         --groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048
    #         --expansion=2 --gating_reduction=8 --drop_rate=0.5
    #         --model=MixNeXtVladModel --num_gpu=4
    #         --batch_size=160 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5
    #         --label_loss=CrossEntropyLoss --eval_data_pattern={datapath}/*.tfrecord --train_dir {train_dir}
    #         --run_once=True
    #         --final_lambda=1 --final_temperature=3
    #         --mix_number=3 --cl_temperature=1 --cl_lambda=1 --diversity=0
    #     """.format(**locals()).split()
    # argv.extend(
    #     options
    # )
    app.run()
