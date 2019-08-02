/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

import com.beust.klaxon.JsonArray
import com.beust.klaxon.Parser
import com.kotlinnlp.utils.ExamplesIndices
import com.kotlinnlp.utils.Shuffler

typealias CharsClassification = List<Int>
typealias AnnotatedSentence = Pair<String, CharsClassification>
typealias Dataset = List<AnnotatedSentence>

/**
 * InvalidDataset Exception.
 */
class InvalidDataset(message: String) : RuntimeException(message)

/**
 * Read a dataset from a JSON file.
 *
 * JSON file format:
 *   - a list of sentences:
 *      - each sentence is a list of 2 elements:
 *        - a string with the text
 *        - a list of integer numbers with the same length of the text, containing the classification of each char
 *
 * Chars classification:
 *   0 = token boundary follows
 *   1 = sentence boundary follows
 *   2 = no boundary follows
 *
 * @param filename the name of the dataset JSON file
 *
 * @return the read [Dataset]
 */
fun readDataset(filename: String): Dataset {

  val examples: JsonArray<*> = Parser().parse(filename) as JsonArray<*>

  @Suppress("UNCHECKED_CAST")
  return examples.map { example -> example as JsonArray<*>

    val sentence: String = example[0] as String
    val classification: CharsClassification = example[1] as CharsClassification

    AnnotatedSentence(sentence, classification)
  }
}

/**
 * A tokenizer dataset that has been merged into a single text.
 *
 * @property fullText the full text obtained merging all the sentences of the dataset
 * @property charsClassification the chars classification of the full text
 */
internal class MergedDataset(val fullText: String, val charsClassification: CharsClassification) {

  /**
   * Check requirements.
   */
  init {
    require(this.fullText.length == this.charsClassification.size)
  }
}

/**
 * Merge a given dataset into a unique text with its chars classification.
 *
 * @param dataset a tokenizer dataset
 *
 * @return a merged dataset
 */
internal fun mergeDataset(dataset: Dataset): MergedDataset {

  val fullText = StringBuffer()
  val fullClassifications = ArrayList<Int>()

  dataset.forEach { (sentence, charsClassification) ->

    if (sentence.length != charsClassification.size)
      throw InvalidDataset("Sentence and chars classification have different lengths")

    fullText.append(sentence)
    charsClassification.forEach { fullClassifications.add(it) }
  }

  return MergedDataset(fullText = fullText.toString(), charsClassification = fullClassifications)
}

/**
 * Shuffle a given dataset.
 *
 * @param dataset a tokenizer dataset
 * @param shuffler the helper to shuffle the dataset
 *
 * @return a new shuffled dataset
 */
internal fun shuffleDataset(dataset: Dataset, shuffler: Shuffler): Dataset {

  val exampleIndices = ExamplesIndices(size = dataset.size, shuffler = shuffler)

  return exampleIndices.map { i -> dataset[i] }
}
