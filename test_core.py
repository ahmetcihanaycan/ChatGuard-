import unittest
from preprocessing import normalize, edit_distance, rule_predict

class PreprocessingTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance('kitten','sitting'),3)
        self.assertEqual(edit_distance('','hello'),5)
    def test_normalization(self):
        self.assertEqual(normalize('  HELLoooo @person https://example.com  '),'helloo user url')
        self.assertEqual(normalize('stup1d'),'stupid')
        self.assertEqual(normalize('score 123'),'score 123')
    def test_rules(self):
        self.assertEqual(rule_predict('You are stup1d'),1)
        self.assertEqual(rule_predict('Thank you for helping'),0)
        self.assertEqual(rule_predict('That is stupud'),1)

if __name__=='__main__':unittest.main()
