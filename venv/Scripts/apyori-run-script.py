#!C:\Users\xlott\PycharmProjects\MarketBasketAnalysis\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'apyori==1.1.2','console_scripts','apyori-run'
__requires__ = 'apyori==1.1.2'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('apyori==1.1.2', 'console_scripts', 'apyori-run')()
    )
