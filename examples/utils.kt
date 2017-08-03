/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.jsoniter.JsonIterator
import java.io.BufferedInputStream
import java.io.FileInputStream

/**
 *
 */
fun readDataset(filename: String): Pair<ArrayList<String>, ArrayList<ArrayList<Int>>> {

  val iterator = JsonIterator.parse(BufferedInputStream(FileInputStream(filename)), 2048)
  val sentences = ArrayList<String>()
  val goldClassifications = ArrayList<ArrayList<Int>>()

  while (iterator.readArray()) {
    while (iterator.readArray()) {

      sentences.add(iterator.readString())

      goldClassifications.add(ArrayList<Int>())
      iterator.readArray()
      while (iterator.readArray())
        goldClassifications.last().add(iterator.readInt())
    }
  }

  return Pair(sentences, goldClassifications)
}
