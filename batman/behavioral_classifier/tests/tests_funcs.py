from funcs import json_flatten
import unittest


class JsonFlatten(unittest.TestCase):
    """Tests for the JsonFlatten
    
    Author: M Jerge and B King"""
    def test_normal_nest_dict(self):
        #normal case
        temp1 = {
            "0" : 0,
            "1" : 1,
            "nested_dict" : {
                "2" : 2,
                "3" : 3
            }
        }

        temp1_exp = {
            "0_" : 0,
            "1_" : 1,
            "nested_dict_2_" : 2,
            "nested_dict_3_" : 3
        }
        self.assertTrue(json_flatten(temp1), temp1_exp)
    
    def test_normal_with_nested_list(self):
        temp1 = {
            "0" : 0,
            "1" : 1,
            "nested_list" : [2,3]
        }
        temp1_w_dict = {
            "0" : 0,
            "1" : 1,
            "nested_dict" : {
                "2" : 2,
                "3" : 3,
                "nested_list" : [2,3]
            }
        }

        temp1_exp = {
            "0_" : 0,
            "1_" : 1,
            "nested_list_" : [2,3]
        }

        temp1_w_dict_exp = {
            "0_" : 0,
            "1_" : 1,
            "nested_dict_2_" : 2,
            "nested_dict_3_" : 3,
            "nested_dict_nested_list_" : [2,3]
        }

        self.assertTrue(json_flatten(temp1), temp1_exp)
        self.assertTrue(json_flatten(temp1_w_dict), temp1_w_dict_exp)

    def test_normal_dictionary(self):
        normal_dictionary = {
            "0": 0,
            "1": 1, 
            "2": 2
        }

        normal_dictionary_exp = {
            "0_": 0,
            "1_": 1, 
            "2_": 2
        }

        self.assertEqual(json_flatten(normal_dictionary), normal_dictionary_exp)


if __name__ == '__main__':
    unittest.main()