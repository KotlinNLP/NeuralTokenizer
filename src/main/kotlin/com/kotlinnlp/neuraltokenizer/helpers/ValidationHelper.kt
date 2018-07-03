/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.helpers

import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Positionable
import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.utils.*
import com.kotlinnlp.utils.Timer

/**
 * A helper for the validation of a [NeuralTokenizer].
 */
class ValidationHelper(val tokenizer: NeuralTokenizer) {

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
   * Validate the [tokenizer] using the given test dataset.
   *
   * @param testSet the test dataset to validate the [tokenizer]
   *
   * @return CoNLL evaluation statistics
   */
  fun validate(testSet: Dataset): EvaluationStats {

    val timer = Timer()

    val outputSentences: List<Sentence> = this.tokenizer.tokenize(text = mergeDataset(testSet).first)
    val goldSentences: List<Sentence> = this.buildDatasetSentences(testSet)

    println("Elapsed time: %s".format(timer.formatElapsedTime()))

    return EvaluationStats(
      tokens = this.buildMetricStats(
        correct = outputSentences.zip(goldSentences).sumBy {
          this.countSamePositionElements(it.first.tokens, it.second.tokens)
        },
        outputTotal = outputSentences.sumBy { it.tokens.size },
        goldTotal = goldSentences.sumBy { it.tokens.size }
      ),
      sentences = this.buildMetricStats(
        correct = this.countSamePositionElements(outputSentences, goldSentences),
        outputTotal = outputSentences.size,
        goldTotal = goldSentences.size
      )
    )
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
   *
   */
  private fun countSamePositionElements(elements1: List<Positionable>, elements2: List<Positionable>): Int {

    var correct = 0
    var index2 = 0

    elements1.forEach { element1 ->

      while (index2 < elements2.lastIndex && elements2[index2].position.end < element1.position.end) index2++

      if (elements2[index2].position == element1.position) correct++
    }

    return correct
  }

  /**
   * @param correct the number of correct elements
   * @param outputTotal the total amount of elements found
   * @param goldTotal the total amount of gold elements
   *
   * @return CoNLL statistics (precision, recall and F1 score) for a single metric
   */
  private fun buildMetricStats(correct: Int, outputTotal: Int, goldTotal: Int): CoNLLStats {

    val correctDouble: Double = correct.toDouble()

    return CoNLLStats( // skip first field (metrics)
      precision = correctDouble / outputTotal,
      recall = correctDouble / goldTotal,
      f1Score = 2 * correctDouble / (outputTotal + goldTotal))
  }
}
