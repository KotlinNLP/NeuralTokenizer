/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of a [NeuralTokenizer].
 *
 * @property language the language within the [NeuralTokenizer] works. If it matches a managed iso-code, special
 *                    resources will be used for the given language. (Default = unknown)
 * @property maxSegmentSize the max size of the segment of text used as buffer
 * @param charEmbeddingsSize the size of each embeddings associated to each character (default = 30)
 * @param hiddenSize the size of the hidden arrays (the output of each RNN of the [BiRNN]) (default = 50)
 * @param hiddenConnectionType the recurrent connection type of the [BiRNN] (default = RAN)
 */
class NeuralTokenizerModel(
  val language: String = "--",
  val maxSegmentSize: Int = 50,
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
   * The number of adding features (in addition to the embeddings array).
   * They are:
   *  - isLetter
   *  - isDigit
   *  - "end of abbreviation"
   *  - "next end of abbreviation"
   */
  val addingFeaturesSize = 4

  /**
   * The [BiRNN] model of the charsEncoder.
   */
  val biRNN: BiRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = charEmbeddingsSize + addingFeaturesSize,
    hiddenSize = hiddenSize,
    hiddenActivation = Tanh(),
    recurrentConnectionType = hiddenConnectionType)

  /**
   * The model of the boundariesEncoder.
   */
  val boundariesNetworkModel = NeuralNetwork(
    LayerInterface(
      type = LayerType.Input.Dense,
      size = 2 * hiddenSize),
    LayerInterface(
      size = 3,
      activationFunction = Softmax(),
      connectionType = LayerType.Connection.Feedforward)
  )

  /**
   * The embeddings mapped to each character.
   */
  val embeddings = EmbeddingsMap<Char>(size = charEmbeddingsSize)

  /**
   * Language iso-code check.
   */
  init {
    require(this.language.length == 2) { "The language iso-code must be 2 chars long" }
    require(this.language == this.language.toLowerCase()) { "The language iso-code must be lower case" }
  }

  /**
   * Serialize this [BiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @return the String representation of this model with the values of all main parameters
   */
  override fun toString() = """
    - Language: %s
    - BiRNN type: %s
    - BiRNN output size: %d
    - Embeddings size: %d
    - Max segment size: %d
  """
    .trimIndent()
    .format(
      this.language,
      this.biRNN.recurrentConnectionType.name,
      this.biRNN.outputSize,
      this.embeddings.size,
      this.maxSegmentSize)
}
