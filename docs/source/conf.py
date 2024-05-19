# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from sphinx.ext.autodoc import between
sys.path.insert(0, os.path.abspath('../'))

def setup(app):
    app.add_config_value('toc_filter_exclude', [], 'html')
    app.connect('autodoc-process-docstring',
                between('^.*SIGNATURE.*$', exclude=True))
    # app.add_css_file('css/custom.css')
    # app.add_js_file('js/custom.js')

    return app

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Qmiotools'
copyright = '2024, Andrés Gómez (CESGA)'
author = 'Andrés Gómez (CESGA)'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

html_logo = "logo_Qmio_v5-azul-300x80.png"
html_show_sourcelink = False