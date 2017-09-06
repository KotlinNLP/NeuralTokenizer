/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.ScheduledUpdater
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNOptimizer
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer

/**
 * The Optimizer of the sub-networks of the [NeuralTokenizer].
 *
 * @property tokenizer the [NeuralTokenizer] to optimize
 * @param charsEncoderUpdateMethod the update method for the charsEncoder (ADAM is default)
 * @param boundariesClassifierUpdateMethod the update method for the boundariesClassifier (ADAM is default)
 * @param embeddingsUpdateMethod the update method for the embeddings (AdaGrad is default)
 */
class NeuralTokenizerOptimizer(
  val tokenizer: NeuralTokenizer,
  charsEncoderUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.001),
  boundariesClassifierUpdateMethod: UpdateMethod = ADAMMethod(stepSize = 0.001),
  embeddingsUpdateMethod: UpdateMethod = AdaGradMethod(learningRate = 0.01)
) : ScheduledUpdater {

  /**
   * The Optimizer of the charsEncoder of the [tokenizer].
   */
  private val charsEncoderOptimizer = BiRNNOptimizer(
    network = this.tokenizer.charsEncoder.network,
    updateMethod = charsEncoderUpdateMethod)

  /**
   * The Optimizer of the boundariesClassifier of the [tokenizer].
   */
  private val boundariesClassifierOptimizer = NetworkParamsOptimizer(
    neuralNetwork = this.tokenizer.boundariesClassifier.network.network,
    updateMethod = boundariesClassifierUpdateMethod)

  /**
   * The Optimizer of the embeddings vectors of the [tokenizer].
   */
  private val embeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsContainer = this.tokenizer.model.embeddings,
    updateMethod = embeddingsUpdateMethod)

  /**
   * Update the parameters of the neural element associated to this optimizer.
   */
  override fun update() {
    this.charsEncoderOptimizer.update()
    this.boundariesClassifierOptimizer.update()
    this.embeddingsOptimizer.update()
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.charsEncoderOptimizer.newEpoch()
    this.boundariesClassifierOptimizer.newEpoch()
    this.embeddingsOptimizer.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.charsEncoderOptimizer.newBatch()
    this.boundariesClassifierOptimizer.newBatch()
    this.embeddingsOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.charsEncoderOptimizer.newExample()
    this.boundariesClassifierOptimizer.newExample()
    this.embeddingsOptimizer.newExample()
  }

  /**
   * Accumulate the parameters errors into the optimizer.
   *
   * @param segment the segment used for the last backward
   */
  fun accumulateErrors(segment: CharSequence) {

    this.charsEncoderOptimizer.accumulate(this.tokenizer.charsEncoder.getParamsErrors(copy = false))
    this.boundariesClassifierOptimizer.accumulate(this.tokenizer.boundariesClassifier.getParamsErrors(copy = false))

    this.tokenizer.charsEncoder.getInputSequenceErrors(copy = false).forEachIndexed { i, errors ->
      this.embeddingsOptimizer.accumulate(
        embeddingIndex = segment[i].toInt(),
        errors = errors.getRange(0, errors.length - 1)
      )
    }
  }
}
