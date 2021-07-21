%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  _____       ______   ____																		%
% |_   _|     |  ____|/ ____| 		Institute of Embedded Systems												%
%   | |  _ __ | |__  | (___    		Wireless Group														%
%   | | | '_ \|  __|  \___ \   		          Zuercher Hochschule Winterthur												%
%  _| |_| | | | |____ ____) |  		(University of Applied Sciences)												%
% |_____|_| |_|______|_____/  		8401 Winterthur, Switzerland												%
%																						%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialisation of the zhawines document class based on scrreport
% -----------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{zhawines}[2015/02/26 ZHAW InES Standard Report]
\LoadClass[oneside,10pt,parskip=half]{scrreprt}


% Definition of special options for the class
% -----------------------------------------------------------------


\ProcessOptions

% Definition of the used packages (special packages add in teh 
% specific document)
% -----------------------------------------------------------------
\usepackage{graphicx}
\usepackage[german,english]{babel}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{ifthen}
\usepackage{url}
\usepackage{iflang}
\usepackage[utf8]{inputenc}
\usepackage{listings}
\usepackage{color}
\usepackage{scrlayer-scrpage}
\usepackage{mathptmx}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage[absolute]{textpos}
\usepackage{makeidx}
\usepackage{colortbl}
\usepackage{pdflscape}
\usepackage{pdfpages}
\usepackage{tabularx}
\usepackage{lmodern}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{wallpaper} %\ThisTileWallPaper{}
\usepackage[super,square]{natbib} %f�r BibTeX Literaturverzeichnis
\usepackage{amsfonts}
\usepackage{pgfplots}



% Definition of the layout of the document, pagesize and headings
% -----------------------------------------------------------------

\geometry{a4paper, top=30mm, left=30mm, right=25mm, bottom=30mm,headsep=10mm,footskip=10mm}

\renewcommand*{\chapterheadstartvskip}{\vspace{0.5cm}}
%\renewcommand*{\chapterheadendvskip}{\vspace{0.5cm}}

\newcommand{\headings}{
	\pagestyle{scrheadings}
	\pagenumbering{arabic}

	%header / footer 
	\clearscrheadfoot
	\automark[section]{chapter}
	\renewcommand{\chapterpagestyle}{scrheadings}
	\ohead{\headmark}
	\ihead{ZHAW - InES}
	\ifthenelse{\equal{\confidential}{yes}}{\chead{\textcolor{red}{\textnormal{\textbf{\textsf{\Large{CONFIDENTIAL}}}}}}}


	\cfoot{\thepage}
	\ifoot{\the\day.\the\month.\the\year}
	\setfootsepline{0.4pt}
	\setheadsepline{0.4pt}
}


% Redefinition of the \maketitle command to create the ZHAW
% title page, added an email parameter (\email)
% -----------------------------------------------------------------
\newcommand{\Email}{info.ines@zhaw.ch}
\newcommand{\email}[1]{\renewcommand{\Email}{#1}}
\renewcommand{\and}{\\}  % separate the authors with a line break

\renewcommand{\maketitle}{
\thispagestyle{empty}
\pagestyle{empty}

% Place the logo on top left
\begin{flushleft}
	\IfLanguageName{german}
	{\includegraphics[height=30mm]{clsFile/logos/de-zhaw-ines-rgb.png}}
	{\includegraphics[height=30mm]{clsFile/logos/en-zhaw-ines-rgb.png}} 
\end{flushleft}

\vskip 8em
\vskip 3em

% Title of the report
\begin{flushleft}
	{\huge \textbf{\@title}}
	\rule{\textwidth}{0.8mm}
\end{flushleft}

\vskip 5em
\vskip 2em

% School and Institute
\begin{flushleft}
	\LARGE
	\IfLanguageName{german}
	{\textsc{Zürcher Hochschule für Angewandte Wissenschaften}}
	{\textsc{Zurich University of Applied Sciences}} \\
    \vspace*{1cm}
    \Large
    \textsc{Institute of Embedded Systems}
\end{flushleft}

% Authors, Version and Date
\vskip 5em
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{ p{3cm} l }
\IfLanguageName{german}{Autoren}{Authors} & \parbox[t]{5cm}{\@author} \\
\IfLanguageName{german}{Version}{Version} & \parbox[t]{5cm}{\version} \\
\end{tabular}
\vskip 1em
\begin{tabular}{ p{3cm} l }
\IfLanguageName{german}{Letzte Änderung}{Last changes} & \parbox[t]{5cm}{\@date} \\
\end{tabular}

% Second page with the contact information and copyright
\newpage
\begin{flushleft}
\IfLanguageName{german}{\textbf{Urheberrechtsinformationen}}{\textbf{Copyright Information}}\newline
This document is the property of the Zurich University of Applied Sciences in Winterthur, Switzerland: All rights reserved. No part of this document may be used or copied in any way without the prior written permission of the Institute.
\newline
\newline
\IfLanguageName{german}{\textbf{Kontakt Adresse}}{\textbf{Contact Information}}\newline
c/o Inst. of Embedded Systems (InES)\newline
Zürcher Hochschule für Angewandte Wissenschaften \newline
Technikumstrasse 22\newline
CH-8401 Winterthur\newline
\newline
Tel.: +41 (0)58 934 75 25\newline
Fax.: +41 (0)58 935 75 25\newline
\newline
E-Mail: \Email\newline
Homepage: \url{http://www.ines.zhaw.ch}\newline
\end{flushleft}
\pagebreak
}



% Listing
% -----------------------------------------------------------------

%listing colors
\definecolor{lgray}{RGB}{250,250,250}
\definecolor{lgreen}{RGB}{63,127,95}
\definecolor{lred}{RGB}{127,0,85}
\definecolor{lblue}{RGB}{42,0,255}


\lstset{		
		basicstyle=\small\ttfamily,
		frame=single,
		numbers=left,	
		numberstyle=\tiny,
		numberblanklines=true,
		captionpos=b,
		extendedchars=true,
		float=ht,
		showtabs=false,
		tabsize=2,
		showspaces=false,
		showstringspaces=false,
		breaklines=true,
		backgroundcolor=\color{lgray},
		keywordstyle=\color{lred}\bfseries, 
		commentstyle=\color{lgreen}\ttfamily,
		stringstyle=\color{lblue},
		xleftmargin=0.5cm,
		xrightmargin=0.5cm
}

\lstloadlanguages{C++}
                                   




