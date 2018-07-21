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
 *
 * @param model the model
 */
class ValidationHelper(model: NeuralTokenizerModel) {

  /**
   * The tokenizer.
   */
  private val tokenizer = NeuralTokenizer(model = model, useDropout = false)

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
    val outputSentences: List<Sentence> = this.tokenizer.tokenize(text = mergeDataset(testSet).first).fixOffset()
    val goldSentences: List<Sentence> = this.buildDatasetSentences(testSet)
    val outputTokens: List<Token> = outputSentences.flatMap { it.tokens }
    val goldTokens: List<Token> = goldSentences.flatMap { it.tokens }

    println("Elapsed time: %s".format(timer.formatElapsedTime()))

    return EvaluationStats(
      tokens = this.buildMetricStats(outputElements = outputTokens, goldElements = goldTokens),
      sentences = this.buildMetricStats(outputElements = outputSentences, goldElements = goldSentences)
    )
  }

  /**
   * Build a sentences list from the given [dataset], like they compose a unique global text.
   *
   * @param dataset a dataset for the [tokenizer]
   *
   * @return a list of [Sentence]s
   */
  private fun buildDatasetSentences(dataset: Dataset): List<Sentence> {

    var start = 0

    return dataset.mapIndexed { i, (text, charsClassification) ->

      val sentence = Sentence(
        position = Position(index = i, start = start, end = start + text.lastIndex),
        tokens = this.buildDatasetTokens(text = text, charsClassification = charsClassification, sentenceStart = start)
      )

      start = sentence.position.end + 1

      sentence
    }
  }

  /**
   * Build the tokens list of a sentence given the [text] and its [charsClassification].
   *
   * @param text the text of a sentence of the dataset
   * @param charsClassification the chars classification of the sentence
   * @param sentenceStart the start index of the sentence in the global text
   *
   * @return a list of [Token]s
   */
  private fun buildDatasetTokens(text: String,
                                 charsClassification: CharsClassification,
                                 sentenceStart: Int): ArrayList<Token> {

    val tokens = ArrayList<Token>()
    var start = 0

    charsClassification.forEachIndexed { i, charClass ->

      if (charClass != 2) { // end of token or end of sentence

        val isSpace: Boolean = start == i && text[i].isWhitespace()

        if (!isSpace)
          tokens.add(Token(
            form = text.substring(start, i + 1),
            position = Position(index = tokens.size, start = sentenceStart + start, end = sentenceStart + i)
          ))

        start = i + 1
      }
    }

    return tokens
  }

  /**
   * Copy this list of sentences, adding an incremental offset to their position, in order to simulate a unique text
   * composed by the sequence of all the sentences.
   *
   * @return a list containing a copy of all the sentences as a unique sequence
   */
  private fun List<Sentence>.fixOffset(): List<Sentence> {

    var offset = 0

    return this.map {

      val curOffset: Int = offset - it.position.start // make the current sentence starting always from 0

      offset += it.position.length

      it.copy(
        position = it.position.copy(start = it.position.start + curOffset, end = it.position.end + curOffset),
        tokens = it.tokens.map {
          it.copy(position = it.position.copy(start = it.position.start + curOffset, end = it.position.end + curOffset))
        }
      )
    }
  }

  /**
   * @param elements1 a list of positionable elements
   * @param elements2 a list of positionable elements
   *
   * @return the count of elements that have the same position within the two lists
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
   * @param outputElements the list of output elements
   * @param goldElements the list of gold elements
   *
   * @return CoNLL statistics (precision, recall and F1 score) about the comparison of the output elements respect to
   *         the gold elements
   */
  private fun buildMetricStats(outputElements: List<Positionable>, goldElements: List<Positionable>): CoNLLStats {

    val correct: Double = this.countSamePositionElements(outputElements, goldElements).toDouble()
    val outputTotal: Int = outputElements.size
    val goldTotal: Int = goldElements.size

    return CoNLLStats( // skip first field (metrics)
      precision = if (outputTotal > 0) correct / outputTotal else 0.0,
      recall = if (goldTotal > 0) correct / goldTotal else 0.0,
      f1Score = if (outputTotal > 0 && goldTotal > 0) 2 * correct / (outputTotal + goldTotal) else 0.0
    )
  }
}
