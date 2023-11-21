# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Margin based AL method.

Samples in batches based on margin scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod


class ConfidenceAL(SamplingMethod):

  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'confidence'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For multiclass classification, must consider the margin between distance for
    top two most likely classes.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """
    probs = model.predict_proba(self.X) 
    uncertainty= probs.max(axis=1)
    srt = np.argsort(uncertainty)
    indices = []
    i = 0
    while len(indices) < N:
        if srt[i] not in already_selected: 
            indices.append(srt[i])
        i += 1
    return indices
