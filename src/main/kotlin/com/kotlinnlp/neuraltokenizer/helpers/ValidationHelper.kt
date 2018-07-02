/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.helpers

import com.kotlinnlp.conllio.CoNLLUEvaluator
import com.kotlinnlp.conllio.CoNLLWriter
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.utils.*

/**
 * A helper for the validation of a [NeuralTokenizer].
 */
class ValidationHelper(val tokenizer: NeuralTokenizer) {

  companion object {

    /**
     * The name of the temporary file in which to write the output of the tokenizer in CoNLL format.
     */
    private val OUTPUT_FILENAME = "/tmp/tokenizer_output_validation_corpus_${System.currentTimeMillis()}.conll"

    /**
     * The name of the temporary file in which to write the test dataset in CoNLL format.
     */
    private val TEST_FILENAME = "/tmp/tokenizer_test_validation_corpus_${System.currentTimeMillis()}.conll"
  }

  /**
   * Statistics given by the CoNLL evaluation script.
   *
   * @property tokens tokens stats from CoNLL evaluation script
   * @property sentences sentences stats from CoNLL evaluation script
   */
  data class EvaluationStats(val tokens: CoNLLStats, val sentences: CoNLLStats)

  /**
   * Statistics given by the CoNLL evaluation script for a single metric.
   *
   * @property precision precision
   * @property recall recall
   * @property f1Score the F1 score
   */
  data class CoNLLStats(val precision: Double, val recall: Double, val f1Score: Double)

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Validate the [tokenizer] using the given test dataset.
   *
   * @param testSet the test dataset to validate the [tokenizer]
   *
   * @return CoNLL evaluation statistics
   */
  fun validate(testSet: Dataset): EvaluationStats {

    this.startTiming()

    val outputSentences: List<Sentence> = tokenizer.tokenize(text = mergeDataset(testSet).first)

    CoNLLWriter.toFile(
      sentences = outputSentences.toCoNLLSentences(),
      outputFilePath = OUTPUT_FILENAME,
      writeComments = false)

    CoNLLWriter.toFile(
      sentences = this.buildDatasetSentences(testSet).toCoNLLSentences(),
      outputFilePath = TEST_FILENAME,
      writeComments = false)

    val evaluation: String? = CoNLLUEvaluator.evaluate(systemFilePath = OUTPUT_FILENAME, goldFilePath = TEST_FILENAME)

    println("Elapsed time: %s".format(this.formatElapsedTime()))

    if (evaluation != null) {

      try {
        return this.extractStats(conllEvaluation = evaluation)
      } catch (e: RuntimeException) {
        throw RuntimeException("Invalid output of the CoNLL evaluation script: $evaluation")
      }

    } else {
      throw RuntimeException("CoNLL evaluation script gave an error")
    }
  }

  /**
   * Build a sentences list from the given [dataset].
   *
   * @param dataset a dataset for the [tokenizer]
   *
   * @return a list of [Sentence]s
   */
  private fun buildDatasetSentences(dataset: Dataset): List<Sentence> =
    dataset.mapIndexed { i, (sentence, charsClassification) ->
      Sentence(
        text = sentence,
        position = Position(index = i, start = 0, end = sentence.lastIndex),
        tokens = this.buildDatasetTokens(sentence = sentence, charsClassification = charsClassification)
      )
    }

  /**
   * Build a tokens list from the given [sentence] string and its [charsClassification].
   *
   * @param sentence a sentence string of the dataset
   *
   * @return a list of [Token]s
   */
  private fun buildDatasetTokens(sentence: String, charsClassification: CharsClassification): ArrayList<Token> {

    val tokens = ArrayList<Token>()
    var startIndex = 0

    charsClassification.forEachIndexed { i, charClass ->

      if (charClass != 2) { // end of token or end of sentence

        val word: String = sentence.substring(startIndex, i + 1)
        val isSpace: Boolean = word.length == 1 && word.toCharArray()[0].isWhitespace()

        if (!isSpace)
          tokens.add(Token(form = word, position = Position(index = tokens.size, start = startIndex, end = i)))

        startIndex = i + 1
      }
    }

    return tokens
  }

  /**
   * @param conllEvaluation the evaluation string given by the CoNLL evaluation script
   *
   * @return CoNLL statistics for Tokens and Sentences
   */
  private fun extractStats(conllEvaluation: String): EvaluationStats {

    val evalLines: List<String> = conllEvaluation.split("\n")

    return EvaluationStats(  // skip first fields (headers)
      tokens = this.extractMetricStats(evalLines[2]),
      sentences = this.extractMetricStats(evalLines[3]))
  }

  /**
   * @param conllEvalLine a line of the output of the CoNLL evaluation script
   *
   * @return CoNLL statistics for a single metric: Precision, Recall and F1 Score
   */
  private fun extractMetricStats(conllEvalLine: String): CoNLLStats {

    val fields: List<String> = conllEvalLine.split("|")

    return CoNLLStats( // skip first field (metrics)
      precision = fields[1].trim().toDouble() / 100.0,
      recall = fields[2].trim().toDouble() / 100.0,
      f1Score = fields[3].trim().toDouble() / 100.0)
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
