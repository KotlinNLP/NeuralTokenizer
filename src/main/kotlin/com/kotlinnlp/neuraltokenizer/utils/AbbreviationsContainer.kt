/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuraltokenizer.utils

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
   * @param resPath the path of the file which contains one abbreviation per line, relative to the resources directory
   *
   * @return the list of abbreviations
   */
  private fun readAbbreviations(resPath: String): Set<String> {

    val lines = this::class.java.getResource(resPath).readText().split("\n")
    val abbreviations = mutableSetOf<String>()

    lines.forEach { line -> abbreviations.add(line) }

    return abbreviations.toSet()
  }

  /**
   * @return the length of the longest string in the set
   */
  private fun Set<String>.getMaxLength(): Int = if (this.isNotEmpty()) this.maxBy { it.length }!!.length else 0
}
