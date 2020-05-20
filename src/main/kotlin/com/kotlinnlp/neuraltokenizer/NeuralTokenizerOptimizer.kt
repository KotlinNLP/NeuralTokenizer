/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * A wrapper of the optimizers of the [NeuralTokenizer] sub-networks.
 *
 * @param charsEncoderUpdateMethod the update method for the charsEncoder (ADAM is default)
 * @param boundariesClassifierUpdateMethod the update method for the boundariesClassifier (ADAM is default)
 * @param embeddingsUpdateMethod the update method for the embeddings (AdaGrad is default)
 */
class NeuralTokenizerOptimizer(
  charsEncoderUpdateMethod: UpdateMethod<*> = RADAMMethod(stepSize = 0.001),
  boundariesClassifierUpdateMethod: UpdateMethod<*> = RADAMMethod(stepSize = 0.001),
  embeddingsUpdateMethod: UpdateMethod<*> = AdaGradMethod(learningRate = 0.01)
) {

  /**
   * The Optimizer of the BiRNN model of the charsEncoder.
   */
  val charsEncoder = ParamsOptimizer(charsEncoderUpdateMethod)

  /**
   * The Optimizer of the model boundariesClassifier.
   */
  val boundariesClassifier = ParamsOptimizer(boundariesClassifierUpdateMethod)

  /**
   * The Optimizer of the embeddings vectors.
   */
  val embeddings = ParamsOptimizer(embeddingsUpdateMethod)

  /**
   * List of all the optimizers.
   */
  val optimizers = listOf(this.charsEncoder, this.boundariesClassifier, this.embeddings)
}
