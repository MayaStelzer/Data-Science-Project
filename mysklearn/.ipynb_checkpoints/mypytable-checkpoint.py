from mysklearn import myutils

"""
Programmer: Maura Sweeney
Class: CPSC 322, Fall 2025
Programming Assignment #6
11/12/25

Description: This program works with 2D tables of data with methods to store, manipulate, clean, join, and summarize data  
"""

import copy
import csv
from tabulate import tabulate
import statistics

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # get column from string name
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Invalid column name {col_identifier}")
            col_index = self.column_names.index(col_identifier)
        
        # get column from int
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier > len(self.column_names):
                raise ValueError(f"Invalid column index {col_identifier}")
            col_index = col_identifier

        col = [row[col_index] for row in self.data]

        if not include_missing_values:
            col = [val for val in col if val != "NA"]
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                try:
                    if val != "NA" or val != None:
                        self.data[i][j] = float(val)
                except (ValueError, TypeError):
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True):
            if 0 <= index < len(self.data):
                self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, "r") as infile:
            reader = csv.reader(infile)
            self.column_names = next(reader)
            self.data = [row for row in reader]
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        seen = {}
        duplicate_indices = []
        key_indices = [self.column_names.index(k) for k in key_column_names]

        for i, row in enumerate(self.data):
            key = tuple(row[index] for index in key_indices)
            if key in seen:
                duplicate_indices.append(i)
            else:
                seen[key] = i
        return duplicate_indices

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        cleaned_data = []
        for row in self.data:
            if all(str(val) != "NA" for val in row):
                cleaned_data.append(row)
        self.data = cleaned_data


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        if col_name not in self.column_names:
            raise ValueError(f"Invalid column name {col_name}")
        index = self.column_names.index(col_name)
        col_values = [row[index] for row in self.data if row[index] != "NA"]
        # If all values are NA, cannot compute average, so return early
        if not col_values:
            return
        avg = sum(col_values) / len(col_values)
        for row in self.data:
            if row[index] == "NA" or None:
                row[index] = avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_table = MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], [])
        
        for col_name in col_names:
            if col_name not in self.column_names:
                continue
            index = self.column_names.index(col_name)
            col = [val for val in self.get_column(index, include_missing_values=False) if val != "NA"]
            if not col:
                continue
            
            col_sorted = sorted(col)
            min_val = min(col_sorted)
            max_val = max(col_sorted)
            mid_val = (min_val + max_val) / 2
            avg_val = sum(col_sorted) / len(col_sorted)
            median_val = statistics.median(col_sorted)
            stats_table.data.append([col_name, min_val, max_val, mid_val, avg_val, median_val])

        return stats_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indices = [self.column_names.index(k) for k in key_column_names]
        other_key_indices = [other_table.column_names.index(k) for k in key_column_names]

        new_header_names = self.column_names[:]
        for col in other_table.column_names:
            if col not in new_header_names:
                new_header_names.append(col)
        
        new_data = []
        for self_row in self.data:
            for other_row in other_table.data:
                # if the values match
                if tuple(self_row[i] for i in self_key_indices) == tuple(other_row[j] for j in other_key_indices):
                    new_row = self_row[:]
                    # add each new value in the row to the new table
                    for k, val in enumerate(other_row):
                        if k not in other_key_indices:
                            new_row.append(val)
                    new_data.append(new_row)

        return MyPyTable(new_header_names, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        self_key_indices = [self.column_names.index(k) for k in key_column_names]
        other_key_indices = [other_table.column_names.index(k) for k in key_column_names]

        new_header_names = self.column_names[:]
        for col in other_table.column_names:
            if col not in new_header_names:
                new_header_names.append(col)

        new_data = []
        self_matched = []
        other_matched = []

        for i, self_row in enumerate(self.data):
            for j, other_row in enumerate(other_table.data):
                # if the values match
                if tuple(self_row[i] for i in self_key_indices) == tuple(other_row[j] for j in other_key_indices):
                    new_row = self_row[:]
                    # add each new value in the row to the new table
                    for k, val in enumerate(other_row):
                        if k not in other_key_indices:
                            new_row.append(val)
                    new_data.append(new_row)
                    self_matched.append(i)
                    other_matched.append(j)
            
        for i, self_row in enumerate(self.data):
            if i not in self_matched:
                new_row = self_row[:]
                for col in other_table.column_names:
                    if col not in key_column_names:
                        new_row.append("NA")
                new_data.append(new_row)

        for j, other_row in enumerate(other_table.data):
            if j not in other_matched:
                new_row = []
                for col in self.column_names:
                    if col in key_column_names:
                        other_index = other_table.column_names.index(col)
                        new_row.append(other_row[other_index])
                    else:
                        new_row.append("NA")
                for k, val in enumerate(other_row):
                    if k not in other_key_indices:
                        new_row.append(val)
                new_data.append(new_row)

        return MyPyTable(new_header_names, new_data)


if __name__ =="__main__":
    t = MyPyTable(["id", "val"], [["A", 1], ["B", "2"], ["C", "NA"], ["D", 4], ["E", "NA"]])
    
    # test get_shape()
    print("Shape: ", t.get_shape())
    print()

    # test get_column()
    print("val: ", t.get_column("val"))
    print("val no NAs:", t.get_column("val", include_missing_values=False))
    print()

    # test convert_to_numeric()
    t.pretty_print()    
    print()
    t.convert_to_numeric()
    t.pretty_print()
    print()

    # test load_from_file()
    t_load = MyPyTable().load_from_file("./test/dummy.csv")
    t_load.pretty_print()
    print()

    # test write_to_file()
    t_load.save_to_file("./test/dummy_out.csv")

    # test drop_rows()
    t.drop_rows([2])  
    t.pretty_print()
    print()

    # test find_duplicates()
    dup_test = MyPyTable(
        ["id", "val"],
        [["X", 1], ["Y", 2], ["X", 1], ["Z", 3]]
    )
    print("duplicates (by id,val):", dup_test.find_duplicates(["id", "val"]))
    print()

    # test remove_rows_with_missing_values()
    t2 = MyPyTable(
        ["id", "val"],
        [["A", 1.0], ["B", "NA"], ["C", 2.0]]
    )
    t2.remove_rows_with_missing_values()
    t2.pretty_print() 
    print()

    # test replace_missing_values_with_column_average()
    t3 = MyPyTable(
        ["id", "val"],
        [["A", 1.0], ["B", "NA"], ["C", 3.0]]
    )
    t3.replace_missing_values_with_column_average("val")
    t3.pretty_print()
    print()

    # test compute_summary_statistics()
    t4 = MyPyTable(
        ["a", "b"],
        [[1, 5], [3, 7], [5, 9]]
    )
    stats = t4.compute_summary_statistics(["a", "b"])
    stats.pretty_print()
    print()

    # test perform_inner_join()
    t_left = MyPyTable(
        ["Product", "Price"],
        [["Apple", 1.0], ["Banana", 0.5], ["Carrot", 0.7]]
    )
    t_right = MyPyTable(
        ["Product", "Quantity"],
        [["Apple", 10], ["Carrot", 20], ["Donut", 5]]
    )
    joined_inner = t_left.perform_inner_join(t_right, ["Product"])
    joined_inner.pretty_print()
    print()

    # test perform_full_outer_join()
    joined_outer = t_left.perform_full_outer_join(t_right, ["Product"])
    joined_outer.pretty_print()
    print()


