/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainer
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.SequenceFeedforwardNetwork
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of a [NeuralTokenizer].
 *
 * @param charEmbeddingsSize the size of each embeddings associated to each character
 * @param hiddenSize the size of the hidden arrays (the output of each RNN of the [BiRNN])
 * @param hiddenConnectionType the recurrent connection type of the [BiRNN] (RAN is default)
 */
class NeuralTokenizerModel(
  charEmbeddingsSize: Int = 30,
  hiddenSize: Int = 100,
  hiddenConnectionType: LayerType.Connection = LayerType.Connection.RAN
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [NeuralTokenizerModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [NeuralTokenizerModel]
     *
     * @return the [NeuralTokenizerModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): NeuralTokenizerModel = Serializer.deserialize(inputStream)
  }

  /**
   * The max number of embeddings into the container.
   */
  private val EMBEDDINGS_COUNT: Int = 1e05.toInt()

  /**
   * The [BiRNN] model of the charsEncoder.
   */
  val biRNN: BiRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = charEmbeddingsSize + 1,
    hiddenSize = hiddenSize,
    hiddenActivation = Tanh(),
    recurrentConnectionType = hiddenConnectionType
  ).initialize()

  /**
   * The [SequenceFeedforwardNetwork] model of the boundariesEncoder.
   */
  val sequenceFeedforwardNetwork = SequenceFeedforwardNetwork(
    inputType = LayerType.Input.Dense,
    inputSize = 2 * hiddenSize,
    outputSize = 3,
    outputActivation = Softmax()
  ).initialize()

  /**
   * The container of embeddings to associate to each character.
   */
  val embeddings = EmbeddingsContainer(
    count = this.EMBEDDINGS_COUNT,
    size = charEmbeddingsSize
  ).randomize()

  /**
   * Serialize this [BiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
