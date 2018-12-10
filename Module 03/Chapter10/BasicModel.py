'''
Created on 01-Sep-2018

@author: Ankit Dixit
'''

# Well first we will import TensorFlow library
import os
import sys

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

import tensorflow as tf

# Here we will define the flags to store our model 
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

# We will import various libraries to save our model
# in proper format
# Name the place holder
placeholder_name = 'a'

# We will do a simple scaler addition
operation_name = 'add'

# Let's create a placeholder to store input number
a = tf.placeholder(tf.int32, name=placeholder_name)

# Input number will be added by a constant 10
b = tf.constant(10)

# Now whenever an input comes we will add constant to that
add = tf.add(a, b, name=operation_name)

# So next step is to create a session which can perform 
# the above task.

# Run a few operations to make sure our model works
with tf.Session() as sess:
    
    #Let's call our model to do an addition
    c = sess.run(add, feed_dict={a: 2})
    print('10 + 2 = {}'.format(c))
    
    # One more time
    d = sess.run(add, feed_dict={a: 10})
    print('10 + 10 = {}'.format(d))
    
    # Pick out the model input and output
    a_tensor = sess.graph.get_tensor_by_name(placeholder_name + ':0')
    sum_tensor = sess.graph.get_tensor_by_name(operation_name + ':0')

    model_input = build_tensor_info(a_tensor)
    model_output = build_tensor_info(sum_tensor)

    # Create a signature definition for tfserving
    signature_definition = signature_def_utils.build_signature_def(
        inputs={placeholder_name: model_input},
        outputs={operation_name: model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    
    # Let's create a model builder which have a specified path 
    # to store our model    
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    
    # Build graph containing tensorflow session which contains all
    # the variables. 
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()