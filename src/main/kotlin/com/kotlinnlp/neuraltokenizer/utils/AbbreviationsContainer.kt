/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

import java.io.File

/**
 * A container of common abbreviations read from a given file.
 *
 * @param resPath the path of the file which contains the abbreviations (one per line)
 */
class AbbreviationsContainer(resPath: String) {

  /**
   * The set of common abbreviations read from the given file.
   */
  val set = this.readAbbreviations(resPath)

  /**
   * The length of the longest abbreviation in the set
   */
  val maxLength: Int = this.set.getMaxLength()

  /**
   * @param filePath the path of the file which contains one abbreviation per line
   *
   * @return the list of abbreviations
   */
  private fun readAbbreviations(filePath: String): Set<String> {

    val file = filePath.getResource()
    val abbreviations = mutableListOf<String>()

    file.forEachLine { line -> abbreviations.add(line) }

    return abbreviations.toSet()
  }

  /**
   * @return the resource file with this [String] as path
   */
  private fun String.getResource(): File {
    return File(Thread.currentThread().contextClassLoader.getResource(this).file)
  }

  /**
   * @return the length of the longest string in the set
   */
  private fun Set<String>.getMaxLength(): Int = if (this.isNotEmpty()) this.maxBy { it.length }!!.length else 0
}
