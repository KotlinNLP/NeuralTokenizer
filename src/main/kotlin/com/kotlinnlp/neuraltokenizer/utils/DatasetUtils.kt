/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

import com.jsoniter.JsonIterator
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import java.io.BufferedInputStream
import java.io.FileInputStream

typealias CharsClassification = ArrayList<Int>
typealias AnnotatedSentence = Pair<String, CharsClassification>
typealias Dataset = ArrayList<AnnotatedSentence>

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

  val iterator = JsonIterator.parse(BufferedInputStream(FileInputStream(filename)), 2048)
  val dataset = Dataset()

  while (iterator.readArray()) {
    while (iterator.readArray()) {

      val sentence: String = iterator.readString()
      val classification = CharsClassification()

      iterator.readArray()
      while (iterator.readArray()) classification.add(iterator.readInt())

      dataset.add(AnnotatedSentence(sentence, classification))
    }
  }

  return dataset
}

/**
 * Merge the given [dataset] into a unique sentence with its chars classification.
 *
 * @param dataset a tokenizer dataset as
 *
 * @return a Pair containing the full text and the corresponding gold classifications
 */
fun mergeDataset(dataset: Dataset): Pair<String, CharsClassification> {

  val fullText = StringBuffer()
  val fullClassifications = ArrayList<Int>()

  dataset.forEach { (sentence, charsClassification) ->

    if (sentence.length != charsClassification.size)
      throw InvalidDataset("Sentence and chars classification have different lengths")

    fullText.append(sentence)
    charsClassification.forEach { fullClassifications.add(it) }
  }

  return Pair(fullText.toString(), fullClassifications)
}

/**
 * Shuffle the given [dataset].
 *
 * @param dataset a tokenizer [Dataset]
 * @param shuffler the [Shuffler] to shuffle the [dataset]
 *
 * @return a new shuffled [Dataset]
 */
fun shuffleDataset(dataset: Dataset, shuffler: Shuffler): Dataset {

  val exampleIndices = ExamplesIndices(size = dataset.size, shuffler = shuffler)

  val shuffledDataset = Dataset()

  exampleIndices.forEach { i -> shuffledDataset.add(dataset[i]) }

  return dataset
}
