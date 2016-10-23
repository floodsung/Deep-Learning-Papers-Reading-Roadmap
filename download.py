import os
import urllib2
import mistune
import bs4 as BeautifulSoup

def download_pdf(link, location, name):
	try:
	    response = urllib2.urlopen(link)
	    file = open(os.path.join(location, name), 'w')
	    file.write(response.read())
	    file.close()
	except urllib2.HTTPError:
		print('>>> Error 404: cannot be downloaded!\n')    

def clean_pdf_link(link):
	if 'arxiv' in link:
		link = link.replace('abs', 'pdf')	
	return link

def clean_header(text):
	return text.replace(' ', '_').replace('/', '_')	   

if __name__ == '__main__':

	with open('README.md') as readme:
		readme_html = mistune.markdown(readme.read())
		readme_soup = BeautifulSoup.BeautifulSoup(readme_html, "html.parser")

	point = readme_soup.find_all('h1')[1]

	while point is not None:
		if point.name == 'h1':
			level1_directory = os.path.join('pdfs', clean_header(point.text))
			os.makedirs(level1_directory)
			print('\n'.join((point.text, "+" * len(point.text), "")))

		elif point.name == 'h2':
			current_directory = os.path.join(level1_directory, clean_header(point.text))
			os.mkdir(current_directory)
			print('\n'.join((point.text, "+" * len(point.text), "")))

		elif point.name == 'p':
			link = clean_pdf_link(point.find('a').attrs['href'])
			extension = os.path.splitext(link)[1][1:]
			name = point.text.split('[' + extension + ']')[0].replace('.', '').replace('/', '_')
			if link is not None:
				print(name + ' (' + link + ')')
				download_pdf(link, current_directory, '.'.join((name, extension)))
		
		point = point.next_sibling			


