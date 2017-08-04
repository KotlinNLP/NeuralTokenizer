/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.simplednn.dataset.Shuffler
import java.io.File
import java.io.FileOutputStream

/**
 * Execute a training of a [NeuralTokenizer] over the training set read from the file given as first argument and save
 * its model into the file given as second argument.
 */
fun main(args: Array<String>) {

  val modelFilename = args[1]

  val tokenizer = NeuralTokenizer(
    model = NeuralTokenizerModel(charEmbeddingsSize = 30, hiddenSize = 100),
    maxSegmentSize = 50)

  TrainingHelper(tokenizer).train(
    trainingSet = readDataset(args[0]),
    batchSize = 100,
    epochs = 15,
    shuffler = Shuffler())

  tokenizer.model.dump(FileOutputStream(File(modelFilename)))
}
