/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.helpers.TrainingHelper
import com.kotlinnlp.neuraltokenizer.utils.readDataset
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.dataset.Shuffler

/**
 * Execute a training of a [NeuralTokenizer] over the training set read from the file given as second argument and save
 * its model into the file given as first argument.
 * If a third filename argument is given, the tokenizer is validated after each epoch over the dataset read from the
 * given file.
 */
fun main(args: Array<String>) {

  val tokenizer = NeuralTokenizer(
    model = NeuralTokenizerModel(
      charEmbeddingsSize = 30,
      hiddenSize = 60,
      hiddenConnectionType = LayerType.Connection.GRU),
    maxSegmentSize = 50)

  val helper = TrainingHelper(
    tokenizer = tokenizer,
    optimizer = NeuralTokenizerOptimizer(
      tokenizer = tokenizer,
      charsEncoderUpdateMethod = ADAMMethod(stepSize = 0.001),
      boundariesClassifierUpdateMethod = ADAMMethod(stepSize = 0.0001),
      embeddingsUpdateMethod = ADAMMethod(stepSize = 0.001)))

  printModel(tokenizer)
  println()

  helper.train(
    trainingSet = readDataset(args[1]),
    batchSize = 100,
    epochs = 30,
    shuffler = Shuffler(),
    validationSet = if (args.size > 2) readDataset(args[2]) else null,
    modelFilename = args[0])
}

/**
 * Print the configuration parameters of the [tokenizer] model.
 *
 * @param tokenizer a [NeuralTokenizer]
 */
private fun printModel(tokenizer: NeuralTokenizer) {

  println("-- MODEL\n")

  println("BiRNN type: %s".format(tokenizer.model.biRNN.recurrentConnectionType))
  println("BiRNN output size: %d".format(2 * tokenizer.model.biRNN.hiddenSize))
  println("Embeddings size: %d".format(tokenizer.model.embeddings.size))
  println("Max segment size: %d".format(tokenizer.maxSegmentSize))
}
