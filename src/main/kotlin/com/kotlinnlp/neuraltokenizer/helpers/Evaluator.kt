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
import com.kotlinnlp.simplednn.helpers.Evaluator
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.stats.MetricCounter
import java.lang.RuntimeException

/**
 * Helper for the evaluation of a [NeuralTokenizerModel].
 *
 * @param model the model to evaluate
 * @param dataset the validation dataset
 */
class Evaluator(
  model: NeuralTokenizerModel,
  private val dataset: Dataset
) : Evaluator<AnnotatedSentence, EvaluationStats>(dataset) {

  /**
   * Overridden for inheritance but not used.
   */
  override val stats: EvaluationStats get() = throw RuntimeException("Not used")

  /**
   * The tokenizer.
   */
  private val tokenizer = NeuralTokenizer(model = model, useDropout = false)

  /**
   * Overridden for inheritance but replaced by the following method.
   */
  override fun evaluate(example: AnnotatedSentence) {}

  /**
   * Evaluate the model.
   *
   * @return the validation statistics
   */
  override fun evaluate(): EvaluationStats {

    val timer = Timer()
    val outputSentences: List<Sentence> = this.tokenizer.tokenize(text = mergeDataset(this.dataset).fullText).fixOffset()
    val goldSentences: List<Sentence> = this.buildDatasetSentences(this.dataset)
    val outputTokens: List<Token> = outputSentences.flatMap { it.tokens }
    val goldTokens: List<Token> = goldSentences.flatMap { it.tokens }

    println("Elapsed time: %s".format(timer.formatElapsedTime()))

    return EvaluationStats(
      tokens = this.buildMetric(outputElements = outputTokens, goldElements = goldTokens),
      sentences = this.buildMetric(outputElements = outputSentences, goldElements = goldSentences)
    ).apply {
      accuracy = tokens.f1Score * Math.pow(sentences.f1Score, 0.5)
    }
  }

  /**
   * Build a sentences list from the given [dataset], such as they compose a unique global text.
   *
   * @param dataset a dataset for the [tokenizer]
   *
   * @return a list of [Sentence]s
   */
  private fun buildDatasetSentences(dataset: Dataset): List<Sentence> {

    var start = 0

    return dataset.mapIndexed { i, (text, charsClassification) ->
      Sentence(
        position = Position(index = i, start = start, end = start + text.lastIndex),
        tokens = this.buildDatasetTokens(text = text, charsClassification = charsClassification, sentenceStart = start)
      ).also {
        start = it.position.end + 1
      }
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
   * composed by the concatenation of all the sentences.
   *
   * @return a list containing a copy of all the sentences as a unique sequence
   */
  private fun List<Sentence>.fixOffset(): List<Sentence> {

    var offset = 0

    return this.map { s ->

      val curOffset: Int = offset - s.position.start // make the current sentence starting always from 0

      offset += s.position.length

      s.copy(
        position = s.position.copy(start = s.position.start + curOffset, end = s.position.end + curOffset),
        tokens = s.tokens.map { t ->
          t.copy(position = t.position.copy(start = t.position.start + curOffset, end = t.position.end + curOffset))
        }
      )
    }
  }

  /**
   * @param outputElements the list of output elements
   * @param goldElements the list of gold elements
   *
   * @return the statistic metrics (precision, recall and F1 score) about the comparison of the output elements respect
   *         to the gold elements
   */
  private fun buildMetric(outputElements: List<Positionable>, goldElements: List<Positionable>): MetricCounter =
    MetricCounter().apply {
      truePos = countSamePositionElements(outputElements, goldElements)
      falsePos = outputElements.size - truePos
      falseNeg = goldElements.size - truePos
    }

  /**
   * @param elements1 a list of positionable elements
   * @param elements2 a list of positionable elements
   *
   * @return the number of elements that have the same position within the two lists (already sorted)
   */
  private fun countSamePositionElements(elements1: List<Positionable>, elements2: List<Positionable>): Int {

    val s1: Set<Pair<Int, Int>> = elements1.asSequence().map { Pair(it.position.start, it.position.end) }.toSet()
    val s2: Set<Pair<Int, Int>> = elements2.asSequence().map { Pair(it.position.start, it.position.end) }.toSet()

    return s1.intersect(s2).size
  }
}
