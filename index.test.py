import unittest
from boddle import boddle
from index import getIndex, predict


class IndexTests(unittest.TestCase):
    def testIndex(self):
        with boddle(params={}):
            self.assertEqual(getIndex(), 'Success')


if __name__ == '__main__':
    unittest.main()
