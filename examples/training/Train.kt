/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.language.getLanguageByIso
import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.helpers.Trainer
import com.kotlinnlp.neuraltokenizer.helpers.Evaluator
import com.kotlinnlp.neuraltokenizer.utils.Dataset
import com.kotlinnlp.neuraltokenizer.utils.readDataset
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.Shuffler
import com.xenomachina.argparser.mainBody

/**
 * Execute the training of a [NeuralTokenizer].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val trainingSet: Dataset = parsedArgs.trainingSetPath.let {
    println("Reading training set from '$it'...")
    readDataset(it)
  }
  val validationSet: Dataset = parsedArgs.validationSetPath.let {
    println("Reading validation set from '$it'...")
    readDataset(it)
  }

  val model = NeuralTokenizerModel(
    language = getLanguageByIso(parsedArgs.langCode.toLowerCase()),
    maxSegmentSize = 50,
    charEmbeddingsSize = 30,
    hiddenSize = 60,
    hiddenConnectionType = LayerType.Connection.GRU)

  val helper = Trainer(
    model = model,
    modelFilename = parsedArgs.modelPath,
    optimizer = NeuralTokenizerOptimizer(
      charsEncoderUpdateMethod = RADAMMethod(stepSize = 0.001),
      boundariesClassifierUpdateMethod = RADAMMethod(stepSize = 0.0001),
      embeddingsUpdateMethod = RADAMMethod(stepSize = 0.001)),
    dataset = trainingSet,
    batchSize = 100,
    epochs = parsedArgs.epochs,
    evaluator = Evaluator(model = model, dataset = validationSet),
    shuffler = Shuffler(),
    useDropout = true)

  println("\n-- MODEL")
  println(model)

  println("\n-- TRAINING")
  helper.train()
}
