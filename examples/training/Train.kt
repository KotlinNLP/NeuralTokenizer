/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.language.getLanguageByIso
import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.helpers.TrainingHelper
import com.kotlinnlp.neuraltokenizer.utils.readDataset
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.Shuffler

/**
 * Execute the training of a [NeuralTokenizer].
 *
 * Command line arguments:
 *   1. The language iso-code ("--" for unknown language).
 *   2. The name of the file in which to save the model.
 *   3. The filename of the training dataset.
 *   4. The filename of the validation set (optional -> if present, the tokenizer is validated on it after each epoch).
 */
fun main(args: Array<String>) {

  val model = NeuralTokenizerModel(
    language = getLanguageByIso(args[0].toLowerCase()),
    maxSegmentSize = 50,
    charEmbeddingsSize = 30,
    hiddenSize = 60,
    hiddenConnectionType = LayerType.Connection.GRU)

  val helper = TrainingHelper(
    tokenizer = NeuralTokenizer(model = model, useDropout = true),
    optimizer = NeuralTokenizerOptimizer(
      model = model,
      charsEncoderUpdateMethod = ADAMMethod(stepSize = 0.001),
      boundariesClassifierUpdateMethod = ADAMMethod(stepSize = 0.0001),
      embeddingsUpdateMethod = ADAMMethod(stepSize = 0.001)))

  println("-- MODEL\n")
  println(model)
  println()

  helper.train(
    trainingSet = readDataset(args[2]),
    batchSize = 100,
    epochs = 30,
    shuffler = Shuffler(),
    validationSet = if (args.size > 2) readDataset(args[3]) else null,
    modelFilename = args[1])
}
