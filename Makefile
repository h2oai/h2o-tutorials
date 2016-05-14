#
# See gitbook command-line docs at:
#
#     https://github.com/GitbookIO/gitbook
#     http://toolchain.gitbook.com/ebook.html
#

build:
	gitbook build
	gitbook pdf ./ ./H2OTutorialsBook.pdf

clean:
	rm -rf _book
	rm -f H2OTutorialsBook.pdf

