/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

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
 * @property biRNN the [BiRNN] model of the charsEncoder
 * @property sequenceFeedforwardNetwork the [SequenceFeedforwardNetwork] model of the boundariesEncoder
 * @property embeddings the container of embeddings to associate to each character
 */
data class NeuralTokenizerModel(
  val biRNN: BiRNN,
  val sequenceFeedforwardNetwork: SequenceFeedforwardNetwork,
  val embeddings: EmbeddingsContainer
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
   * Serialize this [BiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [BiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
