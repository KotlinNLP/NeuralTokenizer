/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.helpers

import com.kotlinnlp.neuraltokenizer.*
import com.kotlinnlp.neuraltokenizer.utils.*
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [NeuralTokenizerModel].
 *
 * @param model the model to train
 * @param modelFilename the name of the file in which to save the serialized model
 * @param dataset the training dataset
 * @param epochs the number of training epochs
 * @param batchSize the size of each batch (default 100)
 * @param optimizer the parameters optimizers wrapper
 * @param evaluator the helper for the evaluation (default null)
 * @param shuffler used to shuffle the examples before each epoch (with pseudo random by default)
 * @param charsDropout the probability of dropout of the chars encoding (default 0.0)
 * @param boundariesDropout the probability of dropout of the boundaries classification (default 0.0)
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class Trainer(
  private val model: NeuralTokenizerModel,
  modelFilename: String,
  private val dataset: Dataset,
  epochs: Int,
  batchSize: Int = 100,
  private val optimizer: NeuralTokenizerOptimizer,
  evaluator: Evaluator,
  private val shuffler: Shuffler = Shuffler(),
  charsDropout: Double = 0.0,
  boundariesDropout: Double = 0.0,
  verbose: Boolean = true
) : Trainer<AnnotatedSentence>(
  modelFilename = modelFilename,
  optimizers = optimizer.optimizers,
  examples = dataset,
  epochs = epochs,
  batchSize = batchSize,
  evaluator = evaluator,
  shuffler = shuffler,
  verbose = verbose
) {

  /**
   * The neural tokenizer built with the given model.
   */
  private val tokenizer = NeuralTokenizer(
    model = this.model,
    charsDropout = charsDropout,
    boundariesDropout = boundariesDropout)

  /**
   * The gold classification of the current segment.
   */
  private lateinit var segmentGoldClassification: List<Int>

  /**
   * The dataset merged into a unique text with its chars classification.
   */
  private lateinit var mergedDataset: MergedDataset

  /**
   * Initialize the Embeddings of the tokenizer model, associating them to the chars contained in the training dataset.
   */
  init {

    this.dataset.forEach { (sentence, _) ->
      sentence.forEach { char ->
        if (char !in this.tokenizer.model.embeddings) {
          this.tokenizer.model.embeddings.set(key = char)
        }
      }
    }
  }

  /**
   * Train the model over an epoch, grouping examples in batches, shuffling them before with the given shuffler.
   */
  override fun trainEpoch() {

    var examplesCount = 0

    this.mergedDataset = mergeDataset(dataset = shuffleDataset(dataset = this.dataset, shuffler = this.shuffler))

    this.newEpoch()

    this.forEachSegment { range ->

      examplesCount++

      this.learnFromExample(range)

      if (examplesCount % this.batchSize == 0)
        this.endOfBatch()
    }

    if (examplesCount % this.batchSize > 0) // last batch in case of remaining examples
      this.endOfBatch()
  }

  /**
   * Iterate over the segments of training extracted shifting the text using the gold chars classifications.
   * Before returning the segment, it is classified from the tokenizer and the [segmentGoldClassification] is set.
   *
   * @param callback a callback called for each segment (it takes the range of segment indices as argument)
   */
  private fun forEachSegment(callback: (IntRange) -> Unit) {

    val textLength: Int = this.mergedDataset.fullText.length
    val progress = ProgressIndicatorBar(total = textLength)
    var startIndex = 0

    while (startIndex < textLength) {

      val segmentRange = IntRange(
        start = startIndex,
        endInclusive = minOf(startIndex + this@Trainer.tokenizer.model.maxSegmentSize, textLength) - 1)

      this.segmentGoldClassification = this.mergedDataset.charsClassification.slice(segmentRange)

      callback(segmentRange)

      val shiftCharIndex = this@Trainer.getShiftCharIndex()

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

    return if (lastSentenceBoundary >= 0)
      lastSentenceBoundary
    else
      this.getMiddleTokenBoundary().let { if (it >= 0) it else segmentSize / 2 }
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
   * Overridden for inheritance but replaced by the following method.
   */
  override fun learnFromExample(example: AnnotatedSentence) {}

  /**
   * Learn from the given segment, comparing its gold classification with the one of the [tokenizer] and accumulate
   * the propagated errors.
   *
   * @param segmentRange the range of char indices of the segment
   */
  private fun learnFromExample(segmentRange: IntRange) {

    this.newExample()

    val segmentClassification: List<DenseNDArray> = this.tokenizer.classifyChars(
      text = this.mergedDataset.fullText,
      start = segmentRange.first,
      length = segmentRange.last - segmentRange.first + 1)

    this.backward(segmentClassification)

    this.accumulateErrors(segment = this.mergedDataset.fullText.subSequence(segmentRange))
  }

  /**
   * Overridden for inheritance but replaced by the following method.
   */
  override fun accumulateErrors() {}

  /**
   * Accumulate the parameters errors into the optimizer.
   *
   * @param segment the segment used for the last backward
   */
  private fun accumulateErrors(segment: CharSequence) {

    this.optimizer.charsEncoder.accumulate(this.tokenizer.charsEncoder.getParamsErrors(copy = false))
    this.optimizer.boundariesClassifier.accumulate(this.tokenizer.boundariesClassifier.getParamsErrors(copy = false))

    this.tokenizer.charsEncoder.getInputErrors(copy = false).forEachIndexed { i, errors ->
      this.optimizer.embeddings.accumulate(
        params = this.tokenizer.model.embeddings[segment[i]],
        errors = errors.getRange(0, errors.length - this.tokenizer.model.addingFeaturesSize)
      )
    }
  }

  /**
   * Catch the event that corresponds to the end of a training batch.
   */
  private fun endOfBatch() {
    this.newBatch()
    this.optimizers.forEach { it.update() }
  }

  /**
   * Backward of the sub-networks of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backward(segmentClassification: List<DenseNDArray>) {

    this.backwardBoundariesClassifier(
      segmentClassification = segmentClassification,
      goldSegmentClassification = this.segmentGoldClassification)

    this.tokenizer.charsEncoder.backward(this.tokenizer.boundariesClassifier.getInputErrors(copy = false))
  }

  /**
   * Backward of the boundaries classifier of the [tokenizer] starting from the errors generated comparing the current
   * [segmentClassification] with the [segmentGoldClassification].
   *
   * @param segmentClassification the classification of the current segment
   */
  private fun backwardBoundariesClassifier(segmentClassification: List<DenseNDArray>,
                                           goldSegmentClassification: List<Int>) {

    this.tokenizer.boundariesClassifier.backward(
      List(
        size = goldSegmentClassification.size,
        init = { i ->
          val charClassification: DenseNDArray = segmentClassification[i]
          val goldClass: Int = goldSegmentClassification[i]

          charClassification[goldClass] = charClassification[goldClass] - 1

          charClassification
        }
      )
    )
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {
    this.tokenizer.model.dump(FileOutputStream(File(modelFilename)))
  }
}
