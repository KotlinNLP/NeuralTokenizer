/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer

import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.progressindicator.ProgressIndicatorBar
import kotlin.coroutines.experimental.buildSequence

/**
 * A helper for the training of a [NeuralTokenizer].
 */
class TrainingHelper(val tokenizer: NeuralTokenizer) {

  /**
   * The [NeuralTokenizerOptimizer] of the [tokenizer].
   */
  private val optimizer = NeuralTokenizerOptimizer(this.tokenizer)

  /**
   * The gold classification of the current segment.
   */
  lateinit private var segmentGoldClassification: ArrayList<Int>

  /**
   * Train the [tokenizer] using the [goldClassifications] as reference of correct predictions.
   *
   * @param sentences the text to tokenize, as list of sentences
   * @param goldClassifications a list containing the correct classifications of the chars of each sentence
   * @param batchSize the size of the training batches (default 1)
   * @param epochs number of epochs
   * @param shuffler the [Shuffler] to shuffle the training sentences before each epoch (default null)
   */
  fun train(sentences: ArrayList<String>,
            goldClassifications: ArrayList<ArrayList<Int>>,
            batchSize: Int = 1,
            epochs: Int = 3,
            shuffler: Shuffler? = null) {

    require(sentences.size == goldClassifications.size)

    println("-- START TRAINING OVER %d SENTENCES".format(sentences.size))

    (0 until epochs).forEach { i ->

      println("Epoch ${i + 1}")

      val mergedSentences = this.mergeSentences(
        sentences = sentences,
        goldClassifications = goldClassifications,
        shuffler = shuffler)

      this.trainEpoch(text = mergedSentences.first, goldClassifications = mergedSentences.second, batchSize = batchSize)
    }
  }

  /**
   * Merge the training sentences into a unique sentences, with the classifications of its chars, shuffling the sentences
   * eventually.
   *
   * @param sentences the list of sentences
   * @param goldClassifications a list containing the correct classifications of the chars of each sentence
   * @param shuffler the [Shuffler] to shuffle the training sentences (default null)
   *
   * @return a Pair containing the full text and the corresponding gold classifications
   */
  private fun mergeSentences(sentences: ArrayList<String>,
                             goldClassifications: ArrayList<ArrayList<Int>>,
                             shuffler: Shuffler?): Pair<String, ArrayList<Int>> {

    val fullText: String
    val fullClassifications = ArrayList<Int>()

    if (shuffler != null) {

      val shuffledSentences = this.shuffleSentences(
        sentences = sentences,
        goldClassifications = goldClassifications,
        shuffler = shuffler)

      fullText = shuffledSentences.first.joinToString("")
      shuffledSentences.second.forEach { classifications -> classifications.forEach { fullClassifications.add(it) } }

    } else {
      fullText = sentences.joinToString("")
      goldClassifications.forEach { classifications -> classifications.forEach { fullClassifications.add(it) } }
    }

    return Pair(fullText, fullClassifications)
  }

  /**
   * Shuffle the training sentences.
   *
   * @param sentences the list of sentences
   * @param goldClassifications a list containing the correct classifications of the chars of each sentence
   * @param shuffler the [Shuffler] to shuffle the training sentences
   *
   * @return a Pair containing the shuffled sentences and the corresponding gold classifications of their chars
   */
  private fun shuffleSentences(sentences: ArrayList<String>,
                               goldClassifications: ArrayList<ArrayList<Int>>,
                               shuffler: Shuffler): Pair<ArrayList<String>, ArrayList<ArrayList<Int>>> {

    val exampleIndices = ExamplesIndices(size = sentences.size, shuffler = shuffler)

    val shuffledSentences = arrayListOf<String>()
    val shuffledClassifications = arrayListOf<ArrayList<Int>>()

    exampleIndices.forEach { i ->
      shuffledSentences.add(sentences[i])
      shuffledClassifications.add(goldClassifications[i])
    }

    return Pair(shuffledSentences, shuffledClassifications)
  }

  /**
   * Train the [tokenizer] over an epoch, using the [goldClassifications] as reference of correct predictions.
   *
   * @param text the text to tokenize
   * @param goldClassifications an array containing the correct classification of each character
   * @param batchSize the size of the training batches
   */
  private fun trainEpoch(text: String, goldClassifications: ArrayList<Int>, batchSize: Int) {

    require(text.length == goldClassifications.size)

    var examplesCount: Int = 0

    this.loopSegments(text = text, goldClassifications = goldClassifications).forEach { (startIndex, endIndex) ->

      examplesCount++

      this.learnFromExample(segment = text.subSequence(startIndex, endIndex + 1))

      if (examplesCount % batchSize == 0) {
        this.endOfBatch()
      }
    }

    if (examplesCount % batchSize > 0) { // last batch with remaining examples
      this.endOfBatch()
    }
  }

  /**
   * Loop over the segments of training extracted shifting the text using the [goldClassifications].
   * Before returning the segment, it is classified from the tokenizer and the [segmentGoldClassification] is set.
   *
   * @param text the text to tokenize
   * @param goldClassifications the gold classification of each character of the [text]
   *
   * @return a Pair containing the start and end indices of the current segment
   */
  private fun loopSegments(text: String, goldClassifications: ArrayList<Int>) = buildSequence {

    val progress = ProgressIndicatorBar(total = text.length)
    var startIndex: Int = 0

    while (startIndex < text.length) {

      val endIndex: Int = minOf(startIndex + this@TrainingHelper.tokenizer.maxSegmentSize - 1, text.lastIndex)

      this@TrainingHelper.segmentGoldClassification = ArrayList(goldClassifications.subList(startIndex, endIndex + 1))

      yield(Pair(startIndex, endIndex))

      val shiftCharIndex = this@TrainingHelper.getShiftCharIndex()

      progress.tick(amount = shiftCharIndex + 1)

      startIndex += shiftCharIndex + 1
    }
  }

  /**
   * @return the index of the last char to remove from the segment shifting it to left
   */
  private fun getShiftCharIndex(): Int {

    val segmentSize : Int = this.segmentGoldClassification.size
    var lastSentenceBoundary: Int = segmentSize - 1

    while (lastSentenceBoundary >= 0 &&
      this.segmentGoldClassification[lastSentenceBoundary] != 1) lastSentenceBoundary--

    return if (lastSentenceBoundary >= 0) {
      lastSentenceBoundary

    } else {
      val middleTokenBoundary: Int = this.getMiddleTokenBoundary()

      if (middleTokenBoundary >= 0) middleTokenBoundary else segmentSize / 2
    }
  }

  /**
   * Return the index of the boundary of the token that crosses the middle of the gold classification of the current
   * segment.
   * Search in the second half of the segment first, starting from the middle. If any boundary is found the search
   * continues in the first half of the segment, starting from the middle again.
   *
   * @return the index of the boundary of the middle token if any, -1 otherwise
   */
  private fun getMiddleTokenBoundary(): Int {

    val segmentSize : Int = this.segmentGoldClassification.size
    val halfSegmentSize: Int = segmentSize / 2
    var middleTokenBoundary: Int = halfSegmentSize

    // search in the second half
    while (middleTokenBoundary < segmentSize &&
      this.segmentGoldClassification[middleTokenBoundary] != 0) middleTokenBoundary++

    if (middleTokenBoundary >= segmentSize) {
      middleTokenBoundary = halfSegmentSize

      // search in the first half
      while (middleTokenBoundary > 0 &&
        this.segmentGoldClassification[middleTokenBoundary] != 0) middleTokenBoundary--
    }

    return if (middleTokenBoundary in 0 until segmentSize) middleTokenBoundary else -1
  }

  /**
   * Learn from the given [segment], comparing its gold classification with the one of the [tokenizer] and accumulate
   * the propagated errors.
   *
   * @param segment the segment of text from which to learn
   */
  private fun learnFromExample(segment: CharSequence) {

    this.optimizer.newExample()

    this.backward(segmentClassification = this.tokenizer.classifyChars(segment))

    this.optimizer.accumulateErrors(segment)
  }

  /**
   * Catch the event that corresponds to the end of a training batch.
   */
  private fun endOfBatch() {
    this.optimizer.newBatch()
    this.optimizer.update()
  }

  /**
   * Backward of the sub-networks of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backward(segmentClassification: Array<DenseNDArray>) {

    this.backwardBoundariesClassifier(
      segmentClassification = segmentClassification,
      goldSegmentClassification = this.segmentGoldClassification)

    this.tokenizer.charsEncoder.backward(
      outputErrorsSequence = this.tokenizer.boundariesClassifier.getInputSequenceErrors(copy = false),
      propagateToInput = true)
  }

  /**
   * Backward of the boundaries classifier of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backwardBoundariesClassifier(segmentClassification: Array<DenseNDArray>,
                                           goldSegmentClassification: ArrayList<Int>) {

    this.tokenizer.boundariesClassifier.backward(
      outputErrorsSequence = Array(
        size = goldSegmentClassification.size,
        init = { i ->
          val charClassification: DenseNDArray = segmentClassification[i]
          val goldClass: Int = goldSegmentClassification[i]

          charClassification[goldClass] = charClassification[goldClass] - 1

          charClassification
        }
      ),
      propagateToInput = true
    )
  }
}
