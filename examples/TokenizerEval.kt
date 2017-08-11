/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.helpers.ValidationHelper
import com.kotlinnlp.neuraltokenizer.utils.readDataset
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [NeuralTokenizer] for the language with the iso-code given as first argument.
 * The model of the tokenizer is read from the serialized file given as second argument.
 * The file given as third argument is used as test set.
 */
fun main(args: Array<String>) {

  println("Reading dataset...")

  val testSet = readDataset(args[2])

  println("Loading model...")

  val tokenizer = NeuralTokenizer(
    language = args[0],
    maxSegmentSize = 50,
    model = NeuralTokenizerModel.load(FileInputStream(File(args[1]))))

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(testSet.size))
  println("Language: ${args[0]}")

  val helper = ValidationHelper(tokenizer)
  val stats: ValidationHelper.EvaluationStats = helper.validate(testSet)

  println()

  println("Tokens accuracy     ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
    .format(100.0 * stats.tokens.precision, 100.0 * stats.tokens.recall, 100.0 * stats.tokens.f1Score))

  println("Sentences accuracy  ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
    .format(100.0 * stats.sentences.precision, 100.0 * stats.sentences.recall, 100.0 * stats.sentences.f1Score))
}
