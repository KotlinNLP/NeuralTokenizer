/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuraltokenizer.*
import java.io.File
import java.io.FileInputStream

/**
 * Execute an evaluation of a [NeuralTokenizer] over the test set read from the file given as second argument.
 * The model of the tokenizer is read from the serialized file given as first argument.
 */
fun main(args: Array<String>) {

  println("Reading dataset...")

  val testSet = readDataset(args[1])

  val tokenizer = NeuralTokenizer(
    maxSegmentSize = 50,
    model = NeuralTokenizerModel.load(FileInputStream(File(args[0]))))

  println("\n-- START VALIDATION OVER %d TEST SENTENCES".format(testSet.size))

  val helper = ValidationHelper(tokenizer)
  val stats: ValidationHelper.EvaluationStats = helper.validate(testSet)

  println()

  println("Tokens accuracy     ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
    .format(100.0 * stats.tokens.precision, 100.0 * stats.tokens.recall, 100.0 * stats.tokens.f1Score))

  println("Sentences accuracy  ->   Precision: %.2f%%  |  Recall: %.2f%%  |  F1 Score: %.2f%%"
    .format(100.0 * stats.sentences.precision, 100.0 * stats.sentences.recall, 100.0 * stats.sentences.f1Score))
}
