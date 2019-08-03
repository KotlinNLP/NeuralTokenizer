/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.helpers.EvaluationStats
import com.kotlinnlp.neuraltokenizer.helpers.ValidationHelper
import com.kotlinnlp.neuraltokenizer.utils.readDataset
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [NeuralTokenizerModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val model = parsedArgs.modelPath.let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(File(it)))
  }

  val validationSet = parsedArgs.validationSetPath.let {
    println("Reading validation set from '$it'...")
    readDataset(it)
  }

  println("\n-- MODEL")
  println(model)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(validationSet.size))
  println("Language: ${model.language}")

  val helper = ValidationHelper(model = model, dataset = validationSet)
  val stats: EvaluationStats = helper.evaluate()

  println("\n-- STATISTICS")
  println(stats)
}
