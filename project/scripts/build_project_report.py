"""Build the report with original covers, real experiment tables and Word fields."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
STAGING=REPO/'.codex-artifacts/report'
STAGING.mkdir(parents=True,exist_ok=True)
ORIGINAL=STAGING/'Nhom05_DoAn_BanGoc.docm'
ASSETS=ROOT/'docs/report-assets'
POINTER=json.loads((ROOT/'artifacts/report-latest.json').read_text(encoding='utf-8'))
RUN=Path(POINTER['directory'])
R=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
CATALOG=json.loads((ROOT/'src/movie_rating/lab_catalog.json').read_text(encoding='utf-8'))
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
REL='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'


def f(value,digits=4):
    if value is None:return 'N/A'
    if isinstance(value,str):return value
    return f'{value:,.{digits}f}'.replace(',','@').replace('.',',').replace('@','.')


def integer(value):return f'{value:,}'.replace(',','.')


def row(algorithm,split='test',group='supervised'):
    return next(r for r in R[group]['rows'] if r['algorithm']==algorithm and r['split']==split)


def field(paragraph,instruction,result=''):
    run=paragraph.add_run()
    begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin')
    code=OxmlElement('w:instrText');code.set(qn('xml:space'),'preserve');code.text=' '+instruction+' '
    sep=OxmlElement('w:fldChar');sep.set(qn('w:fldCharType'),'separate')
    text=OxmlElement('w:t');text.text=result
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end')
    for el in (begin,code,sep,text,end):run._r.append(el)
    return run


def set_font(run,size=13,bold=False):
    run.font.name='Times New Roman';run.font.size=Pt(size);run.font.bold=bold
    fonts=run._r.get_or_add_rPr().get_or_add_rFonts()
    for name in ('ascii','hAnsi','eastAsia','cs'):fonts.set(qn('w:'+name),'Times New Roman')


class Report:
    def __init__(self):
        with ZipFile(ORIGINAL) as z:
            packages={n:z.read(n) for n in z.namelist()}
        types=etree.fromstring(packages['[Content_Types].xml'])
        for el in list(types):
            if el.get('PartName')=='/word/document.xml':el.set('ContentType','application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml')
            if 'vba' in el.get('PartName','').lower():types.remove(el)
        packages['[Content_Types].xml']=etree.tostring(types,xml_declaration=True,encoding='UTF-8',standalone=True)
        rels=etree.fromstring(packages['word/_rels/document.xml.rels'])
        for el in list(rels):
            if 'vba' in el.get('Type','').lower():rels.remove(el)
        packages['word/_rels/document.xml.rels']=etree.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
        # The supplied template has executable ribbon/VBA payloads, unrelated to
        # report content. Never carry them into a generated report.
        for name in list(packages):
            if 'vba' in name.lower() or name.startswith(('customUI/','customXml/')):
                del packages[name]
        for rel_name in ('_rels/.rels','word/_rels/document.xml.rels'):
            clean=etree.fromstring(packages[rel_name])
            for el in list(clean):
                if 'vba' in el.get('Type','').lower() or any(
                    text in el.get('Target','').lower() for text in ('customui/','customxml/')):
                    clean.remove(el)
            packages[rel_name]=etree.tostring(clean,xml_declaration=True,encoding='UTF-8',standalone=True)
        clean_types=etree.fromstring(packages['[Content_Types].xml'])
        for el in list(clean_types):
            if 'vba' in el.get('ContentType','').lower() or any(text in el.get('PartName','').lower() for text in ('vba','customui/','customxml/')):
                clean_types.remove(el)
        packages['[Content_Types].xml']=etree.tostring(clean_types,xml_declaration=True,encoding='UTF-8',standalone=True)
        mem=io.BytesIO()
        with ZipFile(mem,'w',ZIP_DEFLATED) as z:
            for name,data in packages.items():
                if 'vba' not in name.lower():z.writestr(name,data)
        mem.seek(0);self.doc=Document(mem)
        body=self.doc._element.body
        # Body children 0..47 are exactly the user's two original cover pages.
        for el in list(body)[48:]:
            if el.tag!=qn('w:sectPr'):body.remove(el)
        self.chapter=0;self.figures={};self.tables={};self.inventory={'figures':[],'tables':[]}
        self.styles()
        self.configure_section(self.doc.sections[-1],'lowerRoman')

    def styles(self):
        styles=self.doc.styles
        specifications=[('Report Body',13,False,None),('Report Chapter',18,True,0),
                        ('Report Heading 1',14,True,1),('Report Heading 2',13,True,2),
                        ('Report Front',18,True,None),('Report Front TOC',18,True,0),
                        ('Report Figure Caption',13,False,None),('Report Table Caption',13,False,None),
                        ('Report Table Text',12,False,None),('Report Equation',13,False,None)]
        for name,size,bold,outline in specifications:
            style=styles[name] if name in styles else styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
            style.base_style=styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(size)
            style.font.bold=bold;style.font.color.rgb=RGBColor(0,0,0)
            fonts=style.element.get_or_add_rPr().get_or_add_rFonts()
            for attr in ('ascii','hAnsi','eastAsia','cs'):fonts.set(qn('w:'+attr),'Times New Roman')
            pf=style.paragraph_format;pf.line_spacing=1.3;pf.space_before=Pt(6);pf.space_after=Pt(6)
            pf.first_line_indent=Cm(1) if name=='Report Body' else Cm(0)
            pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            if name in ('Report Chapter','Report Front','Report Front TOC','Report Figure Caption','Report Table Caption','Report Equation'):pf.alignment=WD_ALIGN_PARAGRAPH.CENTER
            if outline is not None:
                level=OxmlElement('w:outlineLvl');level.set(qn('w:val'),str(outline));style.element.get_or_add_pPr().append(level)
            if name in ('Report Chapter','Report Front','Report Front TOC'):pf.page_break_before=True
            if name in ('Report Chapter','Report Heading 1','Report Heading 2','Report Front','Report Front TOC','Report Table Caption'):pf.keep_with_next=True
            if name=='Report Table Text':pf.space_before=Pt(0);pf.space_after=Pt(0);pf.first_line_indent=Cm(0)
        for name in ['TOC 1','TOC 2','TOC 3','TOC 4','Table of Figures']:
            style=styles[name] if name in styles else styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
            style.base_style=styles['Report Body'];style.font.name='Times New Roman';style.font.size=Pt(13)
            style.paragraph_format.first_line_indent=Cm(0);style.paragraph_format.space_before=Pt(0);style.paragraph_format.space_after=Pt(0)
        update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');self.doc.settings.element.append(update)

    def configure_section(self,section,number_format='decimal'):
        section.page_width=Cm(21);section.page_height=Cm(29.7)
        section.left_margin=Cm(3.5);section.right_margin=Cm(2.5)
        section.top_margin=Cm(2.5);section.bottom_margin=Cm(2.5)
        section.footer_distance=Cm(1.25);section.header_distance=Cm(1.25)
        props=section._sectPr
        for el in list(props):
            if el.tag in (qn('w:pgNumType'),qn('w:pgBorders')):props.remove(el)
        number=OxmlElement('w:pgNumType');number.set(qn('w:fmt'),number_format);number.set(qn('w:start'),'1');props.append(number)
        section.footer.is_linked_to_previous=False
        footer_part=section.footer
        for child in list(footer_part._element):footer_part._element.remove(child)
        footer=footer_part.add_paragraph();footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
        footer.paragraph_format.space_before=Pt(0);footer.paragraph_format.space_after=Pt(0)
        set_font(field(footer,'PAGE','1'),12)

    def p(self,text):
        paragraph=self.doc.add_paragraph(text,'Report Body')
        return paragraph

    def heading(self,text,level=1):
        return self.doc.add_paragraph(text,'Report Heading 1' if level==1 else 'Report Heading 2')

    def chapter_title(self,number,text):
        self.chapter=number
        return self.doc.add_paragraph(text.upper(),'Report Chapter')

    def front(self,text,toc=True):
        return self.doc.add_paragraph(text.upper(),'Report Front TOC' if toc else 'Report Front')

    def equation(self,text):
        return self.doc.add_paragraph(text,'Report Equation')

    def bullet(self,text,level=0):
        p=self.doc.add_paragraph(('– ' if level==0 else '+ ')+text,'Report Body')
        p.paragraph_format.left_indent=Cm(.8 if level==0 else 1.3)
        p.paragraph_format.first_line_indent=Cm(-.5)
        return p

    def caption(self,label,title,kind):
        counters=self.figures if kind=='figure' else self.tables
        counters[self.chapter]=counters.get(self.chapter,0)+1
        n=counters[self.chapter];tag=f'{label} {self.chapter}.{n}'
        p=self.doc.add_paragraph(style='Report Figure Caption' if kind=='figure' else 'Report Table Caption')
        p.add_run(f'{label} {self.chapter}.')
        field(p,f'SEQ {"Figure" if kind=="figure" else "Table"}{self.chapter} \\* ARABIC',str(n))
        p.add_run(': '+title)
        self.inventory['figures' if kind=='figure' else 'tables'].append({'number':tag,'title':title})
        return tag

    def table(self,title,headers,rows,widths=None):
        tag=self.caption('Bảng',title,'table') if title else ''
        table=self.doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
        table.style='Table Grid'
        widths=widths or [15/len(headers)]*len(headers)
        for col,width in zip(table.columns,widths):col.width=Cm(width)
        for cell,text,width in zip(table.rows[0].cells,headers,widths):cell.width=Cm(width);cell.text=str(text)
        repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in rows:
            for cell,text,width in zip(table.add_row().cells,values,widths):cell.width=Cm(width);cell.text=str(text)
        for ri,row in enumerate(table.rows):
            cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
            for cell in row.cells:
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
                if ri==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EDEDED');cell._tc.get_or_add_tcPr().append(shade)
                for p in cell.paragraphs:
                    p.style=self.doc.styles['Report Table Text']
                    p.paragraph_format.keep_with_next=len(table.rows)<=6 and ri<len(table.rows)-1
                    for run in p.runs:set_font(run,12,ri==0)
        self.doc.add_paragraph('',style='Report Table Text')
        return tag

    def figure(self,name,title,width=15):
        image=ASSETS/(name+'.png')
        if not image.exists():raise FileNotFoundError(image)
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(0)
        p.add_run().add_picture(str(image),width=Cm(width))
        return self.caption('Hình',title,'figure')

    def save(self):
        self.doc.core_properties.title='Dự đoán đánh giá và gợi ý phim trên MovieLens 25M'
        self.doc.core_properties.subject='Báo cáo đồ án học máy — Nhóm 05'
        self.doc.core_properties.author='Nhóm 05'
        self.doc.core_properties.keywords='MovieLens, temporal evaluation, XGBoost, SVD, NCF'
        memory=io.BytesIO();self.doc.save(memory);memory.seek(0)
        # Keep the macro-free DOCX inside a ZIP while external Word processes
        # on this host are modifying newly created .docx files.
        with ZipFile(STAGING/'report-source.zip','w',ZIP_DEFLATED) as z:
            z.writestr('Nhom05_DoAn.docx',memory.getvalue())
        memory.seek(0)
        with ZipFile(memory) as z:parts={n:z.read(n) for n in z.namelist()}
        types=etree.fromstring(parts['[Content_Types].xml'])
        for el in types:
            if el.get('PartName')=='/word/document.xml':el.set('ContentType','application/vnd.ms-word.document.macroEnabled.main+xml')
        parts['[Content_Types].xml']=etree.tostring(types,xml_declaration=True,encoding='UTF-8',standalone=True)
        assert not any('vba' in name.lower() or name.startswith(('customUI/','customXml/')) for name in parts)
        with ZipFile(STAGING/'Nhom05_DoAn_Sach.docm','w',ZIP_DEFLATED) as z:
            for name,data in parts.items():z.writestr(name,data)
        (STAGING/'inventory.json').write_text(json.dumps(self.inventory,ensure_ascii=False,indent=2),encoding='utf-8')


def front_matter(b):
    p=b.front('LỜI CẢM ƠN');p.paragraph_format.page_break_before=False
    b.p('Nhóm 05 xin gửi lời cảm ơn đến Trường Đại học Công Thương Thành phố Hồ Chí Minh và Khoa Công nghệ thông tin đã tạo môi trường học tập, nghiên cứu và thực hành kiến thức về học máy. Nhóm trân trọng cảm ơn quý thầy cô đã giảng dạy, định hướng phương pháp tiếp cận vấn đề và góp ý trong quá trình thực hiện đồ án.')
    b.p('Nhóm xin ghi nhận đóng góp của GroupLens và Đại học Minnesota trong việc công bố bộ dữ liệu MovieLens phục vụ nghiên cứu. Nhóm cũng cảm ơn cộng đồng phát triển các thư viện mã nguồn mở đã cung cấp công cụ để xây dựng, kiểm chứng và tái lập các thực nghiệm trong báo cáo.')
    b.p('Báo cáo được trình bày trên tinh thần trung thực với số liệu đo được, phân biệt rõ kết quả thực nghiệm, nhận định và những giới hạn còn tồn tại. Nhóm mong nhận được góp ý để tiếp tục hoàn thiện phương pháp đánh giá và hệ thống gợi ý phim.')
    b.front('MỤC LỤC',False)
    field(b.doc.add_paragraph(style='Report Body'),'TOC \\o "1-3" \\h \\z \\u','Mục lục tự động')
    b.front('DANH MỤC CÁC KÝ HIỆU VÀ CHỮ VIẾT TẮT')
    b.table(None,['Ký hiệu','Tiếng Anh','Diễn giải'],[
        ('CF','Collaborative Filtering','Lọc cộng tác'),('NCF','Neural Collaborative Filtering','Lọc cộng tác bằng mạng nơ-ron'),
        ('SVD','Singular Value Decomposition','Phân rã giá trị kỳ dị; phân biệt SVD-CF và TruncatedSVD'),
        ('PCA','Principal Component Analysis','Phân tích thành phần chính'),('RF','Random Forest','Rừng ngẫu nhiên'),
        ('RMSE','Root Mean Squared Error','Căn sai số bình phương trung bình'),('MAE','Mean Absolute Error','Sai số tuyệt đối trung bình'),
        ('AUC','Area Under the ROC Curve','Diện tích dưới đường ROC'),('DBI','Davies–Bouldin Index','Chỉ số đánh giá phân cụm'),
        ('CV','Cross-validation','Đánh giá chéo'),('MLP','Multilayer Perceptron','Mạng truyền thẳng nhiều lớp'),
        ('RL','Reinforcement Learning','Học tăng cường'),('TPE','Tree-structured Parzen Estimator','Bộ lấy mẫu tối ưu siêu tham số'),
        ('CSR','Compressed Sparse Row','Lưu ma trận thưa theo hàng'),('RSS','Resident Set Size','Bộ nhớ vật lý của tiến trình'),
        ('u, i','User, item','Người dùng và phim'),('rᵤᵢ, r̂ᵤᵢ','Observed/predicted rating','Rating thật và rating dự đoán'),
        ('μ, bᵤ, bᵢ','Mean and biases','Trung bình và thiên lệch người dùng/phim'),('α, γ, ε','Learning, discount, exploration','Tham số của Q-Learning')],[2.1,5.5,7.4])
    b.front('DANH MỤC HÌNH')
    field(b.doc.add_paragraph(style='Report Body'),'TOC \\h \\z \\t "Report Figure Caption,1"','Danh mục hình tự động')
    b.front('DANH MỤC BẢNG')
    field(b.doc.add_paragraph(style='Report Body'),'TOC \\h \\z \\t "Report Table Caption,1"','Danh mục bảng tự động')
    section=b.doc.add_section(WD_SECTION_START.NEW_PAGE);b.configure_section(section)
    p=b.chapter_title(0,'MỞ ĐẦU');p.paragraph_format.page_break_before=False
    b.p('Sự gia tăng của danh mục phim và lịch sử tương tác tạo ra nhu cầu hỗ trợ người dùng tìm nội dung phù hợp thay vì chỉ sắp xếp theo mức phổ biến. Dự đoán rating, dự đoán nhãn yêu thích, khám phá nhóm sở thích và lựa chọn chuỗi gợi ý là những bài toán liên quan nhưng có đầu ra, giả định và cách đánh giá khác nhau. Vì vậy, một hệ thống thực nghiệm cần tách rõ mục tiêu của từng nhóm thuật toán trước khi so sánh.')
    b.p('Đồ án nghiên cứu dữ liệu MovieLens 25M, triển khai các nhóm hồi quy, phân loại, phân cụm, luật kết hợp, giảm chiều và học tăng cường; đồng thời bổ sung NCF để đối chiếu với mô hình truyền thống và phân rã ma trận. Mục tiêu là xây dựng quy trình có thể tái lập, báo cáo đủ chất lượng lẫn chi phí tính toán và minh họa luồng gợi ý Top-N trên nền tảng web.')
    b.p('Phạm vi số liệu chính của báo cáo là một mẫu tái lập gồm 300 người dùng và 49.552 ratings thật, được chọn từ dữ liệu MovieLens đã chuẩn bị. Bộ dữ liệu nguồn có hơn 25 triệu ratings, nhưng các kết quả trong báo cáo không được diễn giải thành benchmark toàn bộ nguồn. Mọi bảng kết quả và biểu đồ đều sinh từ artifact của lần chạy theo thời gian; không sử dụng số liệu giả để bổ sung các mục còn thiếu.')
    b.p('Báo cáo gồm bốn chương: Chương 1 trình bày tổng quan đề tài; Chương 2 trình bày lý thuyết và cơ chế thuật toán; Chương 3 mô tả môi trường, kiến trúc và cách cài đặt; Chương 4 trình bày bài toán, thông số, dữ liệu, tiêu chí và kết quả thực nghiệm. Phần cuối tổng hợp kết luận theo từng nhóm, những giới hạn và hướng phát triển.')


def chapter_one(b):
    b.chapter_title(1,'CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI')
    b.heading('1.1. BỐI CẢNH VÀ LÝ DO CHỌN ĐỀ TÀI')
    b.p('Các nền tảng phim trực tuyến thường có danh mục lớn hơn khả năng khám phá thủ công của một người dùng. Chỉ đưa ra danh sách phim được nhiều người xem dễ làm người dùng mới bỏ qua nội dung phù hợp với sở thích riêng. Hệ gợi ý tiếp cận vấn đề này thông qua lịch sử đánh giá, đặc trưng phim và quan hệ giữa những người dùng hoặc bộ phim có hành vi tương tự.')
    b.p('MovieLens là bộ dữ liệu phù hợp cho việc học và kiểm chứng phương pháp vì có rating tường minh, mã người dùng/phim và timestamp. Phiên bản 25M tạo điều kiện quan sát độ thưa, chênh lệch mức độ hoạt động và cold-start. Đề tài khai thác những đặc điểm này để xem xét chất lượng mô hình trong một kịch bản có thứ tự thời gian, thay vì chỉ so các điểm số từ những phép chia ngẫu nhiên thuận lợi hơn [1], [2].')
    b.heading('1.2. MỤC TIÊU VÀ CÂU HỎI NGHIÊN CỨU')
    b.p('Mục tiêu tổng quát là xây dựng hệ thống dự đoán đánh giá và gợi ý phim có quy trình thực nghiệm minh bạch. Hệ thống phải đọc dữ liệu thật, kiểm tra hợp lệ, xây biểu diễn phù hợp, huấn luyện, chọn cấu hình và đánh giá trên dữ liệu giữ lại. NCF được xem là một ứng viên cần kiểm chứng, không được mặc định là tốt hơn mô hình đơn giản.')
    for text in [
        'Trong nhóm dự đoán rating, Linear Regression, XGBoost, SVD-CF hay NCF cân bằng tốt hơn giữa RMSE/MAE và thời gian, bộ nhớ?',
        'Trong bài toán thích/không thích, Logistic Regression và Random Forest có thực sự vượt các quyết định đơn giản như luôn đoán lớp đa số?',
        'K-Means và DBSCAN có tạo được các nhóm sở thích thể loại đủ tách biệt, đủ bao phủ để diễn giải?',
        'Các luật Apriori có đạt Support, Confidence, Lift mong đợi và có ổn định trên những khoảng thời gian giữ lại?',
        'PCA/TruncatedSVD nén được bao nhiêu chiều, mất bao nhiêu thông tin và có giúp dự đoán rating so với Mean/Item-KNN?',
        'Q-Learning có cải thiện reward so với các chiến lược Random/Popularity trong cùng môi trường offline hay chưa?']:b.bullet(text)
    b.heading('1.3. ĐỐI TƯỢNG, PHẠM VI VÀ PHƯƠNG PHÁP')
    b.p('Đối tượng nghiên cứu là các tương tác người dùng–phim có rating từ 0,5 đến 5 sao. Metadata thể loại được dùng như thông tin tĩnh; tên phim dùng để trình bày, không phải nhãn học. Dữ liệu nhân khẩu học không có trong MovieLens 25M và không được suy diễn từ mã người dùng [1].')
    b.p('Phạm vi thuật toán gồm bốn mô hình hồi quy rating, hai mô hình phân loại, hai phương pháp phân cụm, Apriori, PCA/TruncatedSVD và Q-Learning. Các baseline Mean Rating, Item-KNN, Random và Popularity được sử dụng để kiểm tra giá trị tăng thêm. Nghiên cứu không xây mô hình sinh nội dung, không triển khai phân tán và không tuyên bố tác động nhân quả đối với người dùng thật.')
    b.p('Phương pháp thực hiện kết hợp nghiên cứu tài liệu gốc, cài đặt bằng thư viện mở, xây dựng pipeline chống rò rỉ dữ liệu và phân tích artifact. Quy trình chia thời gian, chính sách lọc tương tác ít và cách đo latency được xác định trước khi đọc Test. Các nhận định chọn mô hình dựa trên Validation; Test phục vụ đánh giá cuối và phân tích giới hạn, không dùng để chọn lại cấu hình.')
    b.heading('1.4. YÊU CẦU CHỨC NĂNG VÀ KIẾN TRÚC TỔNG THỂ')
    b.p('Ứng dụng web cho phép tìm phim, xem thể loại, ghi nhận lịch sử rating và nhận danh sách Top-N đã loại các phim được chấm. Phòng thực nghiệm có sidebar sáu nhóm, chọn một hoặc nhiều thuật toán, chỉnh hyperparameters, theo dõi tác vụ nền, đọc lịch sử và xuất CSV/Excel/JSON. Các mốc diễn giải có thể điều chỉnh nhưng không làm thay đổi kết quả đã huấn luyện.')
    b.figure('system_architecture','Kiến trúc dữ liệu, pipeline thực nghiệm và ứng dụng web')
    b.p('Kiến trúc tách lớp dữ liệu, thuật toán, đánh giá, lưu artifact và giao diện. Tác vụ huấn luyện của phòng thực nghiệm chạy qua worker riêng thay vì chặn request giao diện. Bộ chạy phục vụ báo cáo được lưu riêng trong report-temporal để không ghi đè dữ liệu hoặc mô hình đang phục vụ web demo hiện có.')
    b.heading('1.5. ĐÓNG GÓP VÀ NGUYÊN TẮC ĐÁNH GIÁ')
    b.p('Đóng góp của đồ án là tích hợp nhiều bài toán gợi ý trong một quy trình có dấu vết, làm rõ sự khác nhau giữa SVD-CF và SVD giảm chiều, bổ sung NCF với learning curve và xây môi trường Q-Learning từ lịch sử rating. Báo cáo đồng thời minh họa rằng kết quả tốt trên một tiêu chí có thể đi kèm điểm yếu quan trọng trên tiêu chí khác.')
    b.p('Một thuật toán chỉ được gọi là phù hợp trong phạm vi dữ liệu, giao thức và ngân sách cụ thể. Không cộng RMSE, F1, Silhouette và Lift thành điểm chung vì chúng đo các mục tiêu khác nhau. Những kết quả chưa vượt baseline, độ bao phủ thấp hoặc chỉ số không xác định vẫn được giữ lại để phản ánh trung thực quá trình thực nghiệm.')


def chapter_two(b):
    b.chapter_title(2,'CHƯƠNG 2: CƠ SỞ LÝ THUYẾT')
    b.heading('2.1. HỆ GỢI Ý VÀ BIỂU DIỄN DỮ LIỆU')
    b.heading('2.1.1. Rating tường minh và ma trận người dùng–phim',2)
    b.p('Lọc cộng tác khai thác cấu trúc tương tác để dự đoán mức độ phù hợp giữa người dùng và phim. Với phản hồi tường minh, rating biểu thị mức đánh giá đã quan sát; khác với phản hồi ẩn như nhấp chuột hoặc lượt xem. Ma trận R có hàng là người dùng, cột là phim và chỉ một phần nhỏ ô có rating. Tập các ô quan sát được ký hiệu Ω [3].')
    b.equation('R ∈ ℝ^(|U|×|I|),   Ω = {(u, i): có rating quan sát},   sparsity = 1 − |Ω|/(|U|·|I|)')
    b.p('Ô thiếu không phải là rating 0 hoặc bằng chứng người dùng không thích. Khi học SVD-CF/NCF, loss chỉ tính trên tương tác quan sát. Khi xây vector user–genre, rating thiếu cho một genre được điền bằng trung bình rating của người dùng từ phần dữ liệu hợp lệ. Ma trận user–item có thể lưu theo dạng thưa CSR; không tạo ma trận dense toàn bộ MovieLens vì chi phí bộ nhớ lớn.')
    b.heading('2.1.2. Cold-start và rò rỉ dữ liệu',2)
    b.p('Cold-start xuất hiện khi người dùng hoặc phim trong dữ liệu cần dự đoán chưa có đủ lịch sử học. Mô hình nhân tố không thể suy ra trực tiếp embedding chưa từng được tối ưu; mô hình dựa đặc trưng phải dựa vào thống kê hoặc metadata còn khả dụng. Vì vậy, cold-start cần được đo độ bao phủ và có fallback, không chỉ loại khỏi tập đánh giá để làm tăng điểm số.')
    b.p('Rò rỉ dữ liệu xảy ra khi thống kê, scaler, tham số hoặc quyết định cấu hình sử dụng thông tin mà tại thời điểm học chưa được phép biết. Chia thời gian và fit lại preprocessing trong từng fold là hai biện pháp quan trọng. Lọc user/phim dựa trên tổng tương tác tương lai cũng có thể gây rò rỉ; trong đồ án, điều kiện tối thiểu được áp dụng riêng cho từng phần fit [18], [22].')
    b.heading('2.2. NHÓM HỒI QUY VÀ DỰ ĐOÁN RATING')
    b.heading('2.2.1. Linear Regression',2)
    b.p('Hồi quy tuyến tính biểu diễn rating bằng tổ hợp tuyến tính của đặc trưng và một hằng số chặn. Các hệ số được ước lượng bằng cực tiểu tổng bình phương sai số. Trong đề tài, đầu vào không phải mã user/movie như đại lượng liên tục mà là thống kê lịch sử, số lượng tương tác, thể loại và mức lệch sở thích thể loại của người dùng [10].')
    b.equation('r̂ = β₀ + Σⱼ βⱼxⱼ,       β* = arg minᵦ Σₙ(rₙ − r̂ₙ)²')
    b.p('Cơ chế fit gồm xây các đặc trưng từ phần học, loại đóng góp trực tiếp của rating đang làm nhãn khỏi thống kê của chính hàng đó, rồi ước lượng hệ số. Khi dự đoán, thống kê đã học được tra theo ID; trường hợp thiếu lịch sử dùng mức trung bình và cờ unknown. Mô hình dễ diễn giải, huấn luyện nhanh, nhưng khả năng biểu diễn quan hệ phi tuyến phụ thuộc vào việc thiết kế đặc trưng.')
    b.heading('2.2.2. XGBoost Regressor',2)
    b.p('XGBoost kết hợp nhiều cây hồi quy theo boosting: mỗi bước thêm một cây nhằm giảm phần sai số còn lại của tổ hợp trước. Hàm mục tiêu gồm loss dự đoán và thành phần phạt độ phức tạp cây. Khai triển theo gradient và Hessian giúp đánh giá các phép tách; cấu trúc cây có thể học tương tác phi tuyến giữa các đặc trưng [4].')
    b.equation('r̂ᵢ = Σₜ fₜ(xᵢ),       L = Σᵢ ℓ(rᵢ, r̂ᵢ) + Σₜ Ω(fₜ)')
    b.p('Trong cài đặt, objective là bình phương sai số, tree_method dùng histogram, số cây/độ sâu/learning rate là các tham số được kiểm soát. Learning rate nhỏ làm mỗi cây đóng góp ít hơn; cây sâu tăng năng lực nhưng có thể tăng overfitting và chi phí. XGBoost vẫn không tự giải quyết thiếu lịch sử: nếu nhiều đặc trưng của user mới đều quay về fallback, ưu thế phi tuyến có thể bị hạn chế.')
    b.heading('2.2.3. SVD Collaborative Filtering',2)
    b.p('SVD-CF trong đồ án là phân rã nhân tố có bias học trên các rating quan sát, thường được gọi là Funk SVD. Mỗi user/phim có một vector k chiều; tích vô hướng mô tả mức tương thích tiềm ẩn. Thành phần μ, bias user và bias phim tách thiên hướng chấm điểm ra khỏi tương tác sở thích [3].')
    b.equation('r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ')
    b.equation('min Σ_(u,i)∈Ω (rᵤᵢ − r̂ᵤᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖² + bᵤ² + bᵢ²)')
    b.p('SGD cập nhật từng cặp nhân tố/bias theo sai số trên rating đã biết, có regularization để hạn chế nghiệm quá khớp. Tăng số factors giúp mô tả nhiều sở thích hơn nhưng cần đủ tương tác để ước lượng ổn định. Đây không phải phép lấy SVD đầy đủ trên một ma trận đã điền 0; người dùng/phim chưa học phải được dự đoán bằng thành phần bias hoặc trung bình còn khả dụng.')
    b.heading('2.2.4. Neural Collaborative Filtering',2)
    b.p('NCF thay phép tương tác tuyến tính đơn giản bằng mạng học quan hệ giữa embedding user và item. Công trình gốc tập trung phản hồi ẩn; đồ án sử dụng biến thể hồi quy rating tường minh, nên đầu ra là một giá trị liên tục và loss là MSE, không sử dụng trực tiếp mục tiêu nhị phân của bản gốc [5], [21].')
    b.figure('ncf_architecture','Biến thể NCF hồi quy rating được sử dụng trong đồ án')
    b.p('Mạng ghép embedding user, embedding phim và tích từng phần tử của hai vector, sau đó qua lớp ẩn ReLU, dropout và lớp đầu ra tuyến tính. Bias nền được học trước; mạng tối ưu phần dư rating bằng AdamW. Khi suy luận, kết quả cộng lại với bias nền rồi giới hạn trong [0,5; 5]. NCF có thể biểu diễn tương tác phi tuyến, nhưng phụ thuộc mạnh vào lượng lịch sử, cấu hình mạng và mức cold-start.')
    b.equation('z = [eᵤ; eᵢ; eᵤ * eᵢ],       r̂ᵤᵢ = μ + bᵤ + bᵢ + W₂·Dropout(ReLU(W₁z+c₁)) + c₂')
    b.p('Early Stopping theo dõi Validation Loss sau mỗi epoch, lưu checkpoint tốt nhất và dừng khi không cải thiện đủ số epoch patience. Đường Train Loss giảm trong khi Validation Loss đứng yên hoặc tăng là dấu hiệu cần kiểm tra overfitting; khoảng cách giữa hai đường còn có thể chịu ảnh hưởng dịch chuyển phân phối và user mới. Sau chọn epoch, mô hình được học lại trên Train + Validation với ngân sách đã khóa.')
    b.heading('2.3. NHÓM PHÂN LOẠI MỨC ĐỘ YÊU THÍCH')
    b.heading('2.3.1. Logistic Regression',2)
    b.p('Logistic Regression dùng hàm sigmoid biến một điểm tuyến tính thành xác suất lớp Thích. Nhãn được xác định bởi rating ≥3,5; decision threshold của xác suất là một tham số khác, không phải ngưỡng rating này. Mô hình tối ưu log-loss có regularization; chuẩn hóa đặc trưng giúp quá trình tối ưu ổn định hơn [11].')
    b.equation('P(y=1|x) = 1/[1 + exp(−(wᵀx+b))],       ŷ = 1 nếu P(y=1|x) ≥ τ')
    b.p('Tham số C điều khiển mức nghịch đảo regularization trong cài đặt. Ngưỡng τ thấp làm tăng số dự đoán Thích, thường tăng Recall nhưng có thể giảm Precision. Logistic dễ triển khai và dự đoán nhanh, song ranh giới tuyến tính có thể không mô tả đủ quan hệ phức tạp; xác suất cũng cần được kiểm tra khi phân phối dữ liệu thay đổi theo thời gian.')
    b.heading('2.3.2. Random Forest Classifier',2)
    b.p('Random Forest học nhiều cây trên các mẫu bootstrap và sử dụng lựa chọn tập con đặc trưng tại các phép tách. Việc lấy trung bình xác suất từ nhiều cây giúp giảm độ biến động so với một cây riêng lẻ. Số cây, độ sâu tối đa và số mẫu tối thiểu ở lá kiểm soát năng lực và chi phí [6], [12].')
    b.equation('P̂(y=1|x) = (1/T) Σₜ Pₜ(y=1|x)')
    b.p('Rừng cây có thể học tương tác phi tuyến và không đòi hỏi chuẩn hóa như Logistic. Tuy nhiên, nếu điều kiện tối ưu là F1 của lớp phổ biến và ngưỡng quá thấp, mô hình vẫn có thể đi đến chiến lược gần như đoán mọi mẫu là Thích. Vì vậy, số cây nhiều hoặc F1 cao không tự bảo đảm rằng hệ thống phân biệt tốt hai lớp.')
    b.heading('2.4. NHÓM PHÂN CỤM SỞ THÍCH NGƯỜI DÙNG')
    b.heading('2.4.1. Biểu diễn user–genre và K-Means',2)
    b.p('Mỗi user được biểu diễn bởi rating trung bình theo từng genre. Một phim nhiều genre đóng góp rating vào từng genre tương ứng; thành phần thiếu được điền bằng rating trung bình user và vector được StandardScaler biến đổi từ thống kê Train. K-Means tìm k tâm cụm bằng lặp giữa bước gán điểm gần tâm nhất và cập nhật tâm bằng trung bình [13].')
    b.equation('J = Σₖ Σ_(x∈Cₖ) ‖x − μₖ‖²')
    b.p('K-Means phù hợp khi cụm có dạng tương đối gọn trong khoảng cách Euclid. Thuật toán cần chọn k và nhạy với scale, khởi tạo, outlier; trong đồ án dùng nhiều lần khởi tạo và seed cố định. Dù có thể gán nhãn cho mọi user hợp lệ, cụm có Silhouette thấp vẫn không nên được diễn giải như nhóm sở thích tách biệt rõ ràng.')
    b.heading('2.4.2. DBSCAN',2)
    b.p('DBSCAN xác định cụm theo mật độ bằng hai tham số eps và min_samples. Điểm core có đủ láng giềng trong bán kính eps; điểm biên được nối với vùng core, còn điểm không thuộc vùng mật độ được gán noise. Thuật toán có thể tìm hình dạng cụm không cầu và không cần khai báo số cụm trước [7], [14].')
    b.p('Đổi lại, DBSCAN nhạy với eps khi mật độ thay đổi hoặc dữ liệu nhiều chiều. Phiên bản gốc không cung cấp predict chuẩn cho điểm mới. Đồ án công bố phép mở rộng: tìm core Train gần nhất, gán nhãn nếu khoảng cách không vượt eps và gán noise nếu vượt. Silhouette/DBI chỉ tính trên điểm không nhiễu nên luôn phải đọc cùng số điểm còn được đánh giá và tỷ lệ noise.')
    b.heading('2.5. APRIORI VÀ KHAI PHÁ LUẬT KẾT HỢP')
    b.p('Apriori khai thác tính chất tập con: nếu một itemset thường xuyên xuất hiện thì mọi tập con của nó cũng thường xuyên xuất hiện. Thuật toán tạo ứng viên theo độ dài tăng dần và loại ứng viên có tập con không đạt min_support. Từ itemsets đủ phổ biến, các luật A → B được sinh và kiểm tra min_confidence [8], [17].')
    b.equation('support(A→B) = P(A∪B);   confidence(A→B) = P(A∪B)/P(A)')
    b.equation('lift(A→B) = confidence(A→B)/P(B)')
    b.p('Transaction trong đồ án là tập phim user đã chấm ít nhất 4 sao trong Train. Support phản ánh độ phủ, Confidence phản ánh xác suất có điều kiện và Lift so với tần suất nền. Lift >1 biểu thị đồng xuất hiện dương, nhưng luật hiếm có thể có Lift cao không ổn định. Luật kết hợp giúp tạo ứng viên gợi ý có thể giải thích; không chứng minh quan hệ nhân quả hoặc thứ tự xem phim.')
    b.heading('2.6. NHÓM GIẢM CHIỀU DỮ LIỆU')
    b.heading('2.6.1. PCA',2)
    b.p('PCA tìm các hướng trực giao giữ lại phương sai lớn của dữ liệu đã center. Khi chỉ giữ k thành phần, vector gốc được chiếu xuống không gian k chiều và có thể tái tạo gần đúng bằng phép biến đổi ngược. Explained Variance Ratio mô tả phần phương sai giữ lại, còn Reconstruction Error đo sai lệch giữa dữ liệu gốc và dữ liệu tái tạo [15].')
    b.equation('Z = (X − μ)Wₖ;       X̃ = ZWₖᵀ + μ;       MSE₍recon₎ = mean[(X − X̃)²]')
    b.p('Giữ nhiều phương sai không đồng nghĩa dự đoán rating tốt nhất, vì phương sai có thể nằm ở những hướng ít liên quan mục tiêu. Do đó, báo cáo kiểm tra cả nén đặc trưng và một phép dự đoán rating downstream. Mốc 90% phương sai hoặc MSE 0,10 là mục tiêu vận hành của dự án, không phải một chuẩn chung cho mọi bài toán.')
    b.heading('2.6.2. TruncatedSVD và phân biệt với SVD-CF',2)
    b.p('TruncatedSVD xấp xỉ X bằng một số ít singular components và không tự center đầu vào. Đây là phép giảm chiều một ma trận đặc trưng đã có, khác SVD-CF là mô hình học các rating thiếu qua tối ưu trên Ω. Trong thực nghiệm giảm chiều, cả PCA và TruncatedSVD nhận cùng vector user–genre đã chuẩn hóa; vì đầu vào gần zero-mean nên kết quả có thể gần trùng nhau [16].')
    b.equation('X ≈ UₖΣₖVₖᵀ')
    b.p('Khác biệt rất nhỏ do solver không đủ để kết luận một thuật toán có chất lượng cao hơn. Cần ưu tiên số chiều đáp ứng mục tiêu, lỗi tái tạo giữ lại, khả năng diễn giải và chi phí trong cùng điều kiện. Với ma trận đặc trưng nhỏ như 19 genres, lợi thế tính toán của randomized SVD có thể không thể hiện rõ.')
    b.heading('2.7. Q-LEARNING CHO GỢI Ý TUẦN TỰ')
    b.p('Q-Learning ước lượng giá trị của một hành động trong trạng thái, xét cả reward hiện tại và giá trị tương lai. Policy epsilon-greedy cân bằng khám phá và khai thác. Trong môi trường gợi ý, state cần chứa thông tin giúp quyết định diễn biến tiếp theo; action là phim được đưa ra và reward được lấy từ phản hồi lịch sử theo quy tắc đã công bố [9], [20].')
    b.equation('Q(s,a) ← Q(s,a) + α[r + γ maxₐ′ Q(s′,a′) − Q(s,a)]')
    b.p('State của đồ án gồm userId, bitmask phim đã chọn, genre phim trước và phản hồi thích/không thích. Phim đã chọn bị loại khỏi tập action trong episode năm bước. Reward có rating được chuẩn hóa thành (rating − 2,75)/2,25; thiếu rating được quy ước 0. Quy ước 0 là giả định môi trường, không phải rating dự đoán hoặc bằng chứng user không thích.')
    b.p('Random và Popularity là baseline tham chiếu, không phải hai mô hình RL được học. Q-Learning chỉ đạt mục tiêu thực nghiệm nếu reward trên các episode giữ lại vượt baseline trong cùng điều kiện. MovieLens thiếu logging propensity cho hành động chưa quan sát, do đó môi trường chỉ là mô phỏng offline, không cho phép kết luận tác động online hoặc counterfactual đáng tin cậy.')
    b.heading('2.8. ĐÁNH GIÁ, CHỌN CẤU HÌNH VÀ BASELINE')
    b.equation('RMSE = √[(1/n)Σᵢ(rᵢ − r̂ᵢ)²];       MAE = (1/n)Σᵢ|rᵢ − r̂ᵢ|')
    b.equation('Precision = TP/(TP+FP);   Recall = TP/(TP+FN);   F1 = 2PR/(P+R)')
    b.p('AUC-ROC sử dụng xác suất hoặc điểm quyết định, không sử dụng nhãn cứng. Accuracy cần đọc theo phân bố lớp; F1 lớp Thích không phản ánh số TN. Silhouette đo tách biệt hình học, DBI so độ phân tán với khoảng cách cụm. N/A được giữ khi chỉ số không xác định, không thay bằng 0 [22].')
    b.p('GridSearchCV đánh giá từng tổ hợp siêu tham số qua các fold. Đồ án sử dụng năm cửa sổ thời gian mở rộng để phần fit của fold luôn nằm trước phần score; khác với KFold xáo trộn IID. Optuna TPE đề xuất tham số NCF dựa trên các trial trước, sử dụng Validation Loss làm mục tiêu. Mean ± Std của năm fold mô tả biến động theo thời kỳ, không phải khoảng tin cậy 95% [18], [19].')
    b.p('Baseline Mean Rating dự đoán hằng bằng trung bình phần fit; Item-KNN tổng hợp các phim láng giềng trên lịch sử bias residual. Khi một mô hình phức tạp chưa vượt baseline trên Validation, không được dùng kết quả Test thuận lợi để chọn ngược lại. Cân bằng độ chính xác và tài nguyên phải bắt đầu bằng bằng chứng về giá trị tăng thêm thực tế.')


def chapter_three(b):
    b.chapter_title(3,'CHƯƠNG 3: MÔI TRƯỜNG THỰC NGHIỆM')
    b.heading('3.1. PHẦN CỨNG VÀ HỆ ĐIỀU HÀNH')
    runtime=R['runtime']
    b.table('Cấu hình máy thực hiện bộ chạy báo cáo',['Thành phần','Giá trị ghi nhận'],[
        ('CPU','Intel Core i5-12500H thế hệ 12; 12 nhân, 16 bộ xử lý logic'),
        ('RAM',f"{f(runtime['ram_bytes']/2**30,2)} GiB khả dụng ở mức dung lượng vật lý"),
        ('Hệ điều hành',runtime['platform']),('Thiết bị tính toán','CPU; không sử dụng tăng tốc GPU trong lần chạy này'),
        ('Luồng tính toán','XGBoost/RF: n_jobs=2; PyTorch: 4 threads; GridSearchCV: n_jobs=1'),
        ('Bộ đo thời gian','time.perf_counter; giây cho batch và µs/rating cho supervised')],[4.2,10.8])
    b.p('Các mô hình trong cùng bộ chạy được thực hiện trên một máy. Số luồng và phạm vi đo được công bố để người đọc không đồng nhất latency thực nghiệm nhỏ với độ trễ của một hệ thống production. Thời gian dưới mức mili giây có thể chịu ảnh hưởng khởi động thư viện, bộ nhớ cache và lịch điều phối của hệ điều hành; một phép đo không đủ chứng minh chênh lệch tốc độ ổn định.')
    b.heading('3.2. CÔNG NGHỆ VÀ THƯ VIỆN')
    roles={'numpy':'Mảng số và tính toán đặc trưng','pandas':'Đọc, lọc, tổng hợp CSV/Parquet',
           'scikit-learn':'Linear/Logistic, RF, clustering, PCA/SVD, CV, metrics',
           'torch':'Embedding, MLP, AdamW và NCF', 'xgboost':'Gradient boosting cho rating',
           'mlxtend':'Apriori và sinh luật','optuna':'TPE cho NCF','gymnasium':'Môi trường gợi ý tuần tự tự định nghĩa'}
    b.table('Thư viện sử dụng trong lần chạy',['Thư viện','Phiên bản','Vai trò'],
            [('Python',runtime['python'],'Ngôn ngữ và môi trường thực thi')]+[(name,v,roles[name]) for name,v in runtime['libraries'].items()],[3.1,2.3,9.6])
    b.p('FastAPI phục vụ API và ứng dụng web. React, TypeScript và Vite được sử dụng cho giao diện; Matplotlib tạo hình khoa học từ artifact; joblib lưu model. Định dạng Parquet giảm chi phí đọc lặp lại, còn JSON/CSV lưu cấu hình, metric và dấu vết. Báo cáo không khẳng định đã sử dụng MLflow: cơ chế logging thực tế của đề tài là file theo run ID.')
    b.heading('3.3. TỔ CHỨC MÃ NGUỒN VÀ ARTIFACT')
    b.table('Phân tách trách nhiệm triển khai',['Thành phần','Trách nhiệm'],[
        ('data/ và experiment_data.py','Kiểm tra dữ liệu, dữ liệu chuẩn bị, cache và thống kê'),
        ('models/','Các mô hình hồi quy, SVD, NCF, bias và Item-KNN'),
        ('experiment_supervised.py','Pipeline thực nghiệm của giao diện'),
        ('experiment_unsupervised.py','Clustering, Apriori và giảm chiều'),
        ('lab.py và api/lab.py','Worker nền, trạng thái, lưu lịch sử và xuất kết quả'),
        ('scripts/report_temporal_experiments.py','Bộ chạy độc lập: global temporal, GridSearchCV, Optuna, Gymnasium'),
        ('scripts/report_figures.py','Sinh hình PNG/PDF từ results.json, không nhập số minh họa'),
        ('scripts/report_demo.py','Web demo của chính artifact báo cáo; Auto có xét baseline'),
        ('artifacts/report-temporal/<run_id>/','Split Parquet, model, dự đoán Test, results.json, metrics.csv')],[5.2,9.8])
    b.p('Cấu trúc này giúp tránh trộn kết quả có nguồn gốc khác nhau. Phòng thực nghiệm UI hiện có giao thức IID riêng; bộ chạy báo cáo sử dụng global temporal và được lưu tách biệt. Các điểm số IID không được chép sang bảng kết quả chính. Bản báo cáo có thể dựng lại từ mã sinh tài liệu và run ID đã công bố mà không phải nhập tay từng giá trị.')
    b.heading('3.4. QUY TRÌNH XỬ LÝ VÀ KIỂM SOÁT RÒ RỈ')
    b.p('Dữ liệu CSV được kiểm tra kiểu, giá trị thiếu, miền rating, ID hợp lệ và cặp user–movie trùng. Dữ liệu sạch gốc vẫn giữ các rating hợp lệ; chính sách bỏ tương tác quá ít được áp dụng trong phần fit của thí nghiệm, không áp dụng dựa trên toàn bộ lịch sử tương lai. Metadata genres được coi là thông tin tĩnh có sẵn khi dự đoán.')
    b.figure('temporal_protocol','Quy trình chia thời gian, chọn cấu hình, refit và đánh giá giữ lại')
    b.p('Sau lấy mẫu user bằng seed 42, các ratings được sắp theo timestamp. Hai biên được chọn theo mục tiêu 70% và 85%; các dòng có cùng timestamp được giữ chung một phía. Train, Validation và Test có các khoảng thời gian không chồng nhau. Mọi scaler, thống kê user/phim và mô hình phải được fit từ phần dữ liệu được phép ở từng giai đoạn.')
    b.p('Trong nhóm supervised, điều kiện tối thiểu là 5 ratings/user và 3 ratings/phim trong phần fit, lặp đến khi ổn định. Các hàng của Validation/Test không bị loại vì khó dự đoán, nhờ đó chất lượng bao gồm cả cold-start. Khi refit 85%, điều kiện này được tính lại từ Train + Validation thay vì sử dụng một danh sách phim lấy từ Test.')
    b.p('Năm fold dùng cửa sổ mở rộng theo timestamp: phần học của một fold luôn trước phần score và timestamp bằng nhau không bị cắt đôi. Đây là biến thể đánh giá chéo theo thời gian, không phải KFold xáo trộn. Mean ± Std cần được đọc như độ biến động giữa những giai đoạn với kích thước phần học khác nhau, không phải kết quả của năm phép lặp IID độc lập.')
    b.heading('3.5. VẬN HÀNH, XUẤT DỮ LIỆU VÀ KIỂM CHỨNG')
    b.p('Trên giao diện, người dùng chọn nhóm, chọn thuật toán và mở phần cấu hình khi cần. Chạy nền có trạng thái queued/running/completed/failed/cancelled; khi server khởi động lại, job chưa xong được nhận diện gián đoạn. Kết quả đã hoàn thành có thể đọc lại hoặc xuất CSV/Excel/JSON. Việc thu gọn cấu hình không hủy giá trị đã nhập.')
    b.p('Các kiểm thử của dự án kiểm tra chỉ số, chia dữ liệu, hủy tác vụ, lưu/đọc nguyên tử và chống rò rỉ. Kiểm thử trình duyệt xác nhận tương tác của cả sáu nhóm, mở lại lịch sử và responsive sáng/tối. Đối với bộ chạy báo cáo, có thêm kiểm tra thứ tự timestamp của từng fold, miền reward và việc model dự đoán không sử dụng rating của hàng giữ lại làm đặc trưng.')
    b.p('Inference supervised đo cả batch Test và báo thêm µs/rating; phép đo classification của bộ báo cáo bao gồm tạo đặc trưng từ builder đã học, khác quy ước chỉ predict_proba trong UI v3. Kích thước model là số byte pickle trong RAM, không phải GPU RAM. Peak RSS là toàn tiến trình, không đủ để quy riêng mức tăng RAM cho một mô hình.')


def chapter_four(b):
    b.chapter_title(4,'CHƯƠNG 4: THỰC NGHIỆM VÀ SO SÁNH')
    b.heading('4.1. BÀI TOÁN ĐƯỢC GIẢI QUYẾT BỞI TỪNG NHÓM')
    b.table('Ánh xạ nhóm thuật toán với đầu vào, đầu ra và mục tiêu',['Nhóm','Bài toán và đầu ra','Thuật toán'],[
        ('Regression','Dự đoán rating liên tục 0,5–5 sao cho một cặp user–phim','Linear, XGBoost, SVD-CF, NCF'),
        ('Classification','Dự đoán xác suất và nhãn Thích/Không thích; rating ≥3,5 là Thích','Logistic, RF'),
        ('Clustering','Khám phá nhóm user có hồ sơ rating theo genre tương tự','K-Means, DBSCAN'),
        ('Association','Tìm luật phim thường được user thích cùng nhau; rating ≥4 tạo item','Apriori'),
        ('Reduction','Nén vector user–genre, kiểm tra thông tin giữ lại và rating downstream','PCA, TruncatedSVD'),
        ('Reinforcement','Học policy gợi ý 5 phim liên tiếp theo phản hồi lịch sử','Q-Learning; Random/Popularity là baseline')],[3.1,7.3,4.6])
    b.p('Các nhóm không cùng tối ưu một đại lượng. Regression dự đoán điểm số; Classification dự đoán nhãn yêu thích, không trực tiếp trả rating; Clustering/Association hỗ trợ diễn giải và tạo ứng viên; Reduction nghiên cứu biểu diễn; RL xem xét chuỗi quyết định. Do đó, so sánh chỉ có ý nghĩa trong cùng nhóm, giao thức và tập đánh giá.')
    b.heading('4.2. THÔNG SỐ THỰC NGHIỆM CHUNG')
    b.table('Thiết lập và quy tắc chung trước khi đọc Test',['Nội dung','Thiết lập'],[
        ('Random seed','42 cho lấy mẫu, khởi tạo mô hình và bộ lấy mẫu Optuna'),
        ('Chia dữ liệu','Global chronological, mục tiêu 70/15/15; giữ nguyên nhóm timestamp bằng nhau'),
        ('Cross-validation','5 fold expanding-window trên Train 70%; không xáo trộn thời gian'),
        ('Lọc cold-start trong fit','Ít nhất 5 ratings/user, 3 ratings/phim; tính lại riêng từng phần fit'),
        ('Chọn hyperparameters','GridSearchCV cho sklearn/SVD wrapper; Optuna TPE 8 trials cho NCF'),
        ('Chọn cấu hình cuối','RMSE Validation thấp nhất cho Regression; F1 lớp Thích Validation cao nhất cho Classification'),
        ('Early Stopping NCF','Tối đa 12 epochs, patience=3; chọn epoch trên Validation; refit không nhìn Test'),
        ('Refit supervised','Train + Validation; áp dụng lại chính sách lọc chỉ trên 85% này'),
        ('Báo cáo final','Test chỉ đánh giá cuối; không điều chỉnh cấu hình sau khi xem Test'),
        ('Log kết quả',f"Run {R['run_id']}; JSON, CSV, dự đoán .npy và model .joblib"),
        ('Mốc mong đợi','Mục tiêu vận hành của dự án; không phải chứng nhận phổ quát')],[4.1,10.9])
    b.p('GridSearchCV được chạy trên các fold thời gian để sàng lọc không gian đã công bố. Mỗi cấu hình ứng viên sau đó được fit trên toàn Train và đánh giá Validation; lựa chọn cuối theo chỉ số Validation phù hợp nhóm. Các điểm CV trong bảng là của cấu hình được chọn và có tính điều kiện sau chọn, không phải một ước lượng nested-CV không chệch.')
    b.p('Với NCF, tám trial Optuna dùng cùng Train/Validation; TPE có ba trial khởi đầu trước khi sử dụng thông tin các trial trước. Trial tốt nhất xác định hyperparameters và epoch dừng. CV của NCF dùng ngân sách tối đa 12 epochs không sử dụng Validation bên ngoài fold; mô hình cuối dùng số epoch tốt nhất đã chọn trên Validation.')
    b.heading('4.3. THỐNG KÊ DATASET VÀ ĐỘ BAO PHỦ')
    source=json.loads((REPO/'dataset_cleaned/cleaning_report.json').read_text(encoding='utf-8'))
    sample=R['provenance']['sample'];prepared=R['provenance']['prepared']
    b.table('Phân biệt dữ liệu nguồn, dữ liệu chuẩn bị và mẫu thực nghiệm',['Thuộc tính','Nguồn MovieLens','Prepared','Mẫu báo cáo'],[
        ('Ratings',integer(25000095),integer(prepared['ratings']),integer(sample['ratings'])),
        ('Users',integer(162541),integer(prepared['users']),integer(sample['users'])),
        ('Phim có ratings',integer(59047),integer(prepared['movies']),integer(sample['movies'])),
        ('Phim trong catalog',integer(62423),integer(62423),integer(sample['catalog_movies'])),
        ('Genres hợp lệ','19','19',str(sample['genres'])),
        ('Rating min/max','0,5 / 5','0,5 / 5','0,5 / 5'),
        ('Rating mean',f(source['files']['ratings.csv']['mean_rating']),f(prepared['rating_mean']),f(sample['rating_mean'])),
        ('Sparsity',f(100*(1-25000095/(162541*59047)),3)+'%',f(prepared['sparsity']*100,3)+'%',f(sample['sparsity']*100,3)+'%')],[4.2,3.8,3.5,3.5])
    b.p('Dữ liệu nguồn gồm ratings.csv, movies.csv, tags.csv và links.csv. Bản làm sạch giữ 25.000.095 ratings hợp lệ, xử lý các trường văn bản và loại các tag không hợp lệ/trùng theo cleaning_report. Prepared giữ trọn lịch sử của một mẫu user 2%; mẫu báo cáo tiếp tục chọn 300 user bằng seed 42. Không được gọi kết quả trên 49.552 ratings là kết quả huấn luyện toàn bộ 25M.')
    b.figure('rating_distribution','Phân bố rating trong mẫu thực nghiệm 300 người dùng')
    parts=R['provenance']['splits']
    date=lambda value:__import__('datetime').datetime.fromtimestamp(value,__import__('datetime').UTC).strftime('%d/%m/%Y')
    b.table('Số lượng và thời gian của các tập trước lọc fit',['Tập','Ratings','Users / phim','Khoảng thời gian UTC'],[
        (s.title(),integer(v['ratings']),f"{v['users']} / {integer(v['movies'])}",f"{date(v['min_timestamp'])}–{date(v['max_timestamp'])}") for s,v in parts.items()],[2.4,2.6,4.2,5.8])
    b.p('Số bản ghi thực tế có thể lệch nhẹ tỷ lệ đích vì không tách các rating có cùng timestamp. Train kết thúc trước Validation và Validation kết thúc trước Test, kể cả timestamp tại hai biên. Sự chênh lệch mức độ hoạt động và số user giữa các khoảng là đặc điểm của mẫu và được giữ nguyên thay vì cân bằng bằng thông tin tương lai.')
    filt=R['provenance']['fit_filter']
    b.table('Ảnh hưởng lọc fit và cold-start',['Đại lượng','Giá trị'],[
        ('Train trước / sau lọc',f"{integer(parts['train']['ratings'])} / {integer(filt['train_retained'])}"),
        ('Rating bị loại khỏi phần fit Train',integer(filt['train_removed'])),
        ('Số rating dùng refit supervised',integer(filt['refit_retained'])),
        ('Validation có user đã biết từ Train',f(parts['validation']['known_user_fraction']*100,2)+'%'),
        ('Test có user đã biết từ Train',f(parts['test']['known_user_fraction']*100,2)+'%'),
        ('Test có user đã biết sau refit 85%',f(filt['test_known_user_refit']*100,2)+'%'),
        ('Test có phim đã biết sau refit 85%',f(filt['test_known_item_refit']*100,2)+'%')],[10.2,4.8])
    b.figure('cold_start_coverage','Độ bao phủ user/phim theo tập, tham chiếu phần fit Train 70%')
    b.p('Cold-start cao là thách thức chính của giao thức global temporal trong mẫu user. Các mốc user/phim đã biết trước và sau refit được phân biệt để tránh quy sai mọi lỗi Test cho một mô hình. Hàng giữ lại không bị loại vì user/phim mới; các fallback do đó có ảnh hưởng lớn đến cả RMSE và latency.')
    b.heading('4.4. TIÊU CHÍ ĐÁNH GIÁ VÀ GIÁ TRỊ MONG ĐỢI')
    for group in CATALOG['groups']:
        rows=[]
        for metric in group['metrics']:
            reference=metric['reference'];value=reference['value']
            op={'lte':'≤','gte':'≥','gt':'>'}[reference['operator']]
            expected=f"{op}{f(value,2)} {reference['unit']}"
            if reference['kind']=='baseline':expected='Δ>0 so với max(Random, Popularity) cùng split'
            if reference['kind']=='budget':expected+='; ngân sách, không kết luận chất lượng'
            rows.append((metric['name'].replace(' ↑','').replace(' ↓',''),metric['definition'],expected))
        if group['id']=='classification':rows.extend([
            ('Precision lớp Thích','Độ đúng trong những dự đoán Thích','Mốc tham khảo ≥0,70; đọc cùng Recall'),
            ('Recall lớp Thích','Tỷ lệ phim Thích thật được phát hiện','Mốc tham khảo ≥0,70; đọc cùng Precision')])
        b.table('Tiêu chí của nhóm '+group['title'],['Tên tiêu chí','Mô tả','Giá trị mong đợi'],rows,[3.8,6.0,5.2])
    b.p('Các mốc được đặt trước như mục tiêu thử nghiệm: RMSE 1 sao và MAE 0,75 sao giới hạn sai lệch rating; Accuracy/F1/AUC 0,70 làm mức sàn ban đầu; Silhouette 0,50 và DBI 1 là heuristic khám phá. Accuracy vượt mốc vẫn chưa đủ nếu không vượt nền lớp đa số. Lift =1 là mốc độc lập, không được đánh giá là liên kết dương [22], [17].')
    b.p('Mốc giữ 90% phương sai và MSE tái tạo 0,10 đánh giá nén trong không gian đã chuẩn hóa, không thay RMSE rating. Ngân sách 8 components đánh giá độ gọn; nếu phải dùng nhiều chiều hơn để giữ thông tin thì cần nêu sự đánh đổi. Training time 60 giây và inference 0,10 giây/batch chỉ có ý nghĩa khi giữ cùng số mẫu, máy và cách đo.')
    b.p('Support/Confidence là độ phủ và xác suất có điều kiện, không chứng nhận luật hữu ích nếu Lift không vượt 1. Reward Q-Learning phải vượt baseline cùng môi trường; reward thô dương nhưng thua Popularity vẫn chưa đạt mục tiêu cải thiện. N/A được giữ là không xác định, không thay bằng 0 để tạo thứ hạng.')
    regression_results(b);classification_results(b);clustering_results(b);association_results(b);reduction_results(b);rl_results(b);deployment_results(b)


def regression_results(b):
    b.heading('4.5. THỰC NGHIỆM REGRESSION VÀ DỰ ĐOÁN RATING')
    configs=R['supervised']['best_configs'];cv=R['supervised']['cv']
    b.heading('4.5.1. Cấu hình được thử và được chọn',2)
    b.table('Không gian tìm kiếm của bốn mô hình dự đoán rating',['Mô hình','Không gian / thiết lập','Cấu hình chọn'],[
        ('Linear','Bình phương tối thiểu; đặc trưng user/phim/genre','Cấu hình mặc định'),
        ('XGBoost','100 cây; depth ∈{3,6}; learning rate 0,05; reg_lambda=10',f"depth={configs['xgboost']['max_depth']}; 100 cây; lr=0,05"),
        ('SVD-CF','factors ∈{8,16}; 12 epochs; lr=0,005; λ=0,02',f"factors={configs['svd']['factors']}; 12 epochs"),
        ('NCF','Optuna 8 trial; factors {8,16,32}; hidden {16,32}; lr log[0,001;0,01]; λ log[0,0001;0,02]; dropout {0,1;0,2}',
         f"factors={configs['ncf']['factors']}; hidden={configs['ncf']['hidden_size']}; lr={f(configs['ncf']['learning_rate'],6)}; λ={f(configs['ncf']['regularization'],6)}; dropout={f(configs['ncf']['dropout'],1)}; epoch={configs['ncf']['epochs']}")],[2.6,6.4,6.0])
    b.p('Linear/XGBoost nhận đặc trưng thống kê và genres; SVD/NCF chỉ nhận tương tác rating qua các ID danh mục. Vì các đầu vào không hoàn toàn đồng nhất, kết quả thể hiện hiệu quả của những pipeline cụ thể trong đồ án chứ không phải chứng minh ưu thế kiến trúc khi mọi thông tin đầu vào đã được kiểm soát tuyệt đối.')
    b.heading('4.5.2. Chất lượng qua năm fold và đánh giá cuối',2)
    names={'linear':'Linear','xgboost':'XGBoost','svd':'SVD-CF','ncf':'NCF'}
    b.table('Regression: Mean ± Std trên năm fold thời gian',['Mô hình','RMSE (sao)','MAE (sao)'],[
        (name,f"{f(cv[a]['metrics']['rmse']['mean'])} ± {f(cv[a]['metrics']['rmse']['std'])}",
         f"{f(cv[a]['metrics']['mae']['mean'])} ± {f(cv[a]['metrics']['mae']['std'])}") for a,name in names.items()],[3.5,5.75,5.75])
    b.table('Regression: Validation trước refit và Test sau refit',['Mô hình','RMSE Val','RMSE Test','MAE Test'],[
        (name,f(row(a,'validation')['rmse']),f(row(a)['rmse']),f(row(a)['mae'])) for a,name in {**names,'mean':'Mean','item_knn':'Item-KNN'}.items()],[3.6,3.8,3.8,3.8])
    b.figure('regression_rmse','RMSE của bốn mô hình: năm fold thời gian và hai tập giữ lại')
    b.figure('regression_mae_baselines','MAE của mô hình học và baseline; mốc 0,75 sao là mục tiêu dự án')
    b.p(f"Trong bốn mô hình học, XGBoost có RMSE Validation thấp nhất ({f(row('xgboost','validation')['rmse'])}), nên là ứng viên học được chọn trước khi mở Test. Trên Test, RMSE của XGBoost là {f(row('xgboost')['rmse'])}, thấp hơn Linear, SVD-CF và NCF trong lần xác nhận này. Tuy nhiên, toàn bộ bốn mô hình đều chưa đạt RMSE ≤1 và MAE ≤0,75 sao; kết luận phù hợp là ứng viên tốt nhất trong nhóm nhưng chất lượng rating còn hạn chế.")
    b.p(f"Mean Rating đạt RMSE Validation {f(row('mean','validation')['rmse'])}, thấp hơn XGBoost. Do quy tắc chọn bằng Validation đã được khóa, không được dùng việc XGBoost có Test tốt hơn Mean để chọn ngược lại sau khi nhìn Test. Khi có xét baseline, phương án mặc định an toàn của demo báo cáo là Mean/Popularity; XGBoost được giữ như ứng viên ML và chế độ minh họa cá nhân hóa, chưa được tuyên bố có giá trị tăng thêm ổn định.")
    b.p('Linear có bias biểu diễn thấp nhưng khó học phi tuyến khi nhiều user mới phải dùng thống kê mặc định. SVD cần lịch sử để học nhân tố; user/phim mới làm dự đoán quay về bias. NCF còn có nhiều tham số hơn và chưa tận dụng được mạng tương tác ở phần lớn trường hợp cold-start. Đây là các cơ chế giải thích khả dĩ được hỗ trợ bởi độ bao phủ; dữ liệu hiện tại không tách riêng từng nguyên nhân bằng ablation nên không khẳng định quan hệ nhân quả.')
    b.heading('4.5.3. Chi phí tính toán và learning curve NCF',2)
    b.table('Chi phí fit cuối và suy luận của mô hình rating',['Mô hình','Fit (s)','µs/rating','Model (KiB)'],[
        (name,f(row(a)['fit_seconds'],6),f(row(a)['inference_us_per_row'],3),f(row(a)['model_bytes']/1024,1)) for a,name in names.items()],[3.6,3.8,3.8,3.8])
    b.figure('regression_cost','Chi phí fit cuối và suy luận batch Test trên cùng máy CPU')
    b.p('Fit cuối không bao gồm tổng chi phí GridSearchCV, Optuna và năm fold. Latency thấp của SVD/NCF trong trường hợp này còn chịu ảnh hưởng tỷ lệ cold-start cao: những cặp không đủ embedding bỏ qua một phần tính toán và dùng fallback. Không được suy ra rằng mạng sâu luôn nhanh hơn hồi quy trong mọi luồng suy luận.')
    b.figure('ncf_learning_curve','Train/Validation MSE của trial NCF được chọn; đường dọc là epoch tốt nhất')
    curve=R['supervised']['learning_curves']['ncf']
    b.p(f"NCF được chọn ở epoch {configs['ncf']['epochs']}; quá trình dừng sau epoch {curve[-1]['epoch']} theo patience=3. Train MSE giảm từ {f(curve[0]['train_rating_mse'])} xuống {f(curve[-1]['train_rating_mse'])}, trong khi Validation MSE duy trì quanh {f(curve[0]['validation_rmse']**2,2)}. Khoảng cách lớn cho thấy cải thiện fit Train không chuyển thành lợi ích giữ lại; có thể liên quan đồng thời overfitting, thời gian và người dùng mới. Learning curve không đủ để kết luận nguyên nhân duy nhất.")
    b.p('Chênh lệch giữa RMSE trung bình CV và Test còn cho thấy khả năng tổng quát hóa thay đổi theo giai đoạn. Năm fold sớm hơn có người dùng/phim và kích thước Train khác, nên không thể coi Mean ± Std là sai số đo thuần túy. Muốn chứng minh ưu thế ổn định cần thêm mẫu, rolling backtest và báo cáo warm/cold riêng.')


def classification_results(b):
    b.heading('4.6. THỰC NGHIỆM CLASSIFICATION')
    b.heading('4.6.1. Cấu hình, nhãn và kết quả',2)
    cv=R['supervised']['cv'];configs=R['supervised']['best_configs']
    b.p('Nhãn 0: rating từ 0,5 đến dưới 3,5; nhãn 1: rating từ 3,5 đến 5. Logistic thử C∈{0,1;1}, RF thử depth∈{6;12} với 100 cây và min_samples_leaf=3. Xác suất được kiểm tra các ngưỡng 0,10–0,90 trên Validation; mốc rating 3,5 luôn giữ nguyên, không bị thay bởi ngưỡng xác suất.')
    b.table('Phân bố nhãn theo thời gian',['Tập','Không thích (0)','Thích (1)','Tỷ lệ Thích'],[
        (s.title(),integer(v['negative']),integer(v['positive']),f(v['positive']/(v['positive']+v['negative'])*100,2)+'%') for s,v in R['supervised']['class_balance'].items()],[2.8,4.0,4.0,4.2])
    b.table('Classification: Mean ± Std qua năm fold ở ngưỡng 0,5',['Mô hình','Accuracy','F1 lớp Thích','AUC-ROC'],[
        (name,*[f"{f(cv[a]['metrics'][k]['mean'])} ± {f(cv[a]['metrics'][k]['std'])}" for k in ('accuracy','f1','auc_roc')]) for a,name in [('logistic','Logistic'),('random_forest','RF')]],[2.8,4.0,4.1,4.1])
    b.table('Classification: chất lượng cuối ở ngưỡng đã chọn',['Mô hình / tập','Accuracy','F1','AUC-ROC'],[
        (name+' / '+s.title(),f(row(a,s)['accuracy']),f(row(a,s)['f1']),f(row(a,s)['auc_roc'])) for a,name in [('logistic','Logistic'),('random_forest','RF')] for s in ('validation','test')],[4.2,3.6,3.6,3.6])
    b.figure('classification_quality','Accuracy, F1 và AUC-ROC Test; mốc 0,70 không thay thế baseline')
    b.figure('classification_roc','Đường ROC từ xác suất dự đoán trên Test')
    b.p(f"RF có F1 Validation {f(row('random_forest','validation')['f1'])}, cao hơn Logistic {f(row('logistic','validation')['f1'])}, nên là ứng viên được chọn theo đúng tiêu chí ban đầu. Ngưỡng xác suất chọn cho cả hai là {f(configs['logistic']['threshold'],2)}. Trên Test, F1 của hai mô hình gần nhau; không chọn lại Logistic chỉ vì F1 Test nhỉnh hơn.")
    b.heading('4.6.2. Phân tích baseline và ma trận nhầm lẫn',2)
    b.table('Precision, Recall và chi phí trên Test',['Mô hình','Precision / Recall','Fit (s)','µs/rating'],[
        (name,f"{f(row(a)['precision'])} / {f(row(a)['recall'])}",f(row(a)['fit_seconds'],6),f(row(a)['inference_us_per_row'],3)) for a,name in [('logistic','Logistic'),('random_forest','RF')]],[3.0,5.0,3.5,3.5])
    b.figure('classification_confusion','Confusion Matrix Test: hàng là nhãn thật, cột là nhãn dự đoán')
    balance=R['supervised']['class_balance']['test'];positive=balance['positive']/(balance['positive']+balance['negative']);always_f1=2*positive/(1+positive)
    b.p(f"Baseline luôn đoán Thích trên Test đạt Accuracy {f(positive)} và F1 {f(always_f1)}. RF có Recall=1 và Precision xấp xỉ tỷ lệ Thích, cho thấy ngưỡng 0,10 làm mô hình dự đoán gần như tất cả là Thích. F1 vượt 0,70 vì vậy chưa chứng minh khả năng cá nhân hóa; ở mức quyết định cuối, RF không vượt baseline đơn giản này.")
    b.p(f"AUC Test của RF là {f(row('random_forest')['auc_roc'])}, gần nhưng chưa đạt mốc 0,70; Logistic chỉ đạt {f(row('logistic')['auc_roc'])}. Logistic nhanh và gọn hơn, song khả năng xếp hạng hai lớp trong giai đoạn giữ lại yếu hơn RF. RF giữ được tín hiệu xếp hạng tốt hơn nhưng cần ngưỡng/chi phí lớp phù hợp để biến xác suất thành quyết định có ích.")
    b.p('Kết luận cho nhóm là RF tốt nhất theo F1 Validation trong không gian cấu hình đã thử, nhưng chưa đạt yêu cầu phân biệt thực dụng khi xét đồng thời Accuracy, AUC và baseline. Hướng khắc phục là bổ sung mục tiêu Precision tối thiểu, balanced accuracy/PR-AUC, hiệu chỉnh xác suất và kiểm thử theo thời gian, không điều chỉnh lại trên Test hiện tại.')


def clustering_results(b):
    b.heading('4.7. THỰC NGHIỆM CLUSTERING')
    b.p('Biểu diễn đầu vào có 19 thành phần rating trung bình theo genre. Train chỉ giữ user đủ ít nhất 5 ratings cho hồ sơ phân cụm, điền thành phần thiếu bằng mean user và fit StandardScaler trên Train. K-Means thử K={3,4,6}; DBSCAN thử eps={1;1,5;2} và min_samples={3;5}. Điểm score giữ lại được gán bằng mô hình Train, không refit trên Test.')
    b.table('Cấu hình phân cụm được chọn',['Thuật toán','Cấu hình','Phương pháp gán dữ liệu mới'],[
        ('K-Means',str(R['clustering']['best_configs']['kmeans']),'Tâm Train gần nhất'),
        ('DBSCAN',str(R['clustering']['best_configs']['dbscan']),'Core Train gần nhất trong eps; ngoài eps là noise')],[3.1,4.8,7.1])
    b.table('Chất lượng và độ bao phủ cụm',['Mô hình / tập','Silhouette','DBI','Noise / số cụm'],[
        (r['name']+' / '+r['split'].title(),f(r['silhouette']),f(r['dbi']),f"{f(r['noise_fraction']*100,2)}% / {r['clusters']}") for r in R['clustering']['rows']],[4.5,3.2,3.2,4.1])
    b.figure('clustering_pca','PCA 2D của các cụm Train; màu xám biểu thị noise')
    b.figure('clustering_quality_noise','Silhouette, DBI và noise trên Test phải được diễn giải cùng nhau')
    b.figure('clustering_search','Các cấu hình K và eps đã thử; chỉ số Train hỗ trợ chẩn đoán')
    km=row('kmeans',group='clustering');db=row('dbscan',group='clustering')
    b.p(f"K-Means có Silhouette Validation {f(row('kmeans','validation','clustering')['silhouette'])}, cao hơn DBSCAN {f(row('dbscan','validation','clustering')['silhouette'])}; đồng thời không bỏ user thành noise. Trên Test, K-Means đạt Silhouette {f(km['silhouette'])} và DBI {f(km['dbi'])}. Phương án này phù hợp hơn để phân đoạn đầy đủ người dùng trong hai ứng viên hiện tại, nhưng chưa đạt mục tiêu Silhouette 0,50/DBI 1.")
    b.p(f"DBSCAN có DBI Test thấp ({f(db['dbi'])}) nhưng noise {f(db['noise_fraction']*100,2)}%. Với 45 user Test, phần không nhiễu chỉ khoảng ba user chia thành hai cụm; DBI đẹp vì đánh giá trên một tập rất nhỏ không đủ để kết luận DBSCAN tốt hơn. Sự khác biệt mật độ theo thời gian và eps cố định là các giả thuyết giải thích cần được kiểm tra tiếp.")
    profiles=[]
    for cluster in R['clustering']['clusters']:
        if cluster['algorithm']=='kmeans':
            for p in cluster['profiles']:
                profiles.append((str(p['cluster']),str(p['users']),' ; '.join(f"{g}: {f(v,2)}" for g,v in p['genre_means'].items())))
    b.table('Các genre có rating trung bình cao trong từng cụm K-Means Train',['Cụm','Users','Genre và rating trung bình'],profiles,[1.6,2.1,11.3])
    b.p('Bảng genre giúp diễn giải tương đối trong mẫu, không đặt tên cứng cho nhân khẩu học hay bản chất của user. Những khác biệt có thể còn phản ánh thiên hướng chấm cao/thấp thay vì sở thích thể loại riêng. Có thể tiếp tục thử chuẩn hóa theo user, tăng mẫu và đánh giá độ ổn định cụm qua nhiều thời kỳ.')


def association_results(b):
    b.heading('4.8. THỰC NGHIỆM ASSOCIATION RULES VỚI APRIORI')
    r=R['association']['rows'][0]
    b.p('Apriori là thuật toán học chính duy nhất của nhóm này. Transaction gồm phim rating ≥4 trong Train; giữ cả user có transaction rỗng để mẫu số support phản ánh toàn bộ user Train. Chỉ xét 80 phim thích phổ biến nhất Train, min_support=0,08, min_confidence=0,30 và itemset dài tối đa hai phim. Vì vậy, kết luận đánh giá dựa trên tiêu chí và luật có ý nghĩa, không tạo một cuộc thi giả giữa nhiều tên thuật toán.')
    b.table('Tổng hợp kết quả Apriori trên Train',['Đại lượng','Giá trị'],[
        ('Transactions',integer(r['transactions'])),('Frequent itemsets',integer(r['itemsets'])),
        ('Luật đạt ngưỡng',integer(r['rule_count'])),('Support trung bình',f(r['support'])),
        ('Confidence trung bình',f(r['confidence'])),('Lift trung bình',f(r['lift'])),
        ('Execution time (s)',f(r['fit_seconds'],6)),('Inference chuẩn','Không áp dụng; không tạo latency giả')],[9.8,5.2])
    top=R['association']['rules'][:5]
    b.table('Năm luật có Lift cao, đối chiếu với biểu đồ',['Luật và cặp phim','Support','Confidence','Lift'],[
        (f"{i+1}. "+' + '.join(t['antecedents'])+' → '+' + '.join(t['consequents']),f(t['support']),f(t['confidence']),f(t['lift'])) for i,t in enumerate(top)],[7.2,2.6,2.6,2.6])
    b.figure('association_lift','Lift của năm luật trong bảng; đường tham chiếu 1 biểu thị mức độc lập')
    b.figure('association_support_confidence','Phân bố Support–Confidence của toàn bộ luật; màu là Lift')
    b.p(f"Apriori sinh {integer(r['rule_count'])} luật; Support và Confidence trung bình vượt ngưỡng sàng lọc, Lift trung bình {f(r['lift'])}>1. Các luật giữa những phần phim hoặc các phim gần phong cách có thể tạo ứng viên gợi ý dễ giải thích. Tuy nhiên, đây là thống kê được chọn trên Train và bị ảnh hưởng bởi catalog top 80; không được coi số luật lớn là đồng nghĩa chất lượng cao.")
    b.table('Đánh giá cùng tập luật cố định trên dữ liệu giữ lại',['Tập','Support','Confidence','Lift','Phủ vế trái'],[
        (v['split'].title(),f(v['support']),f(v['confidence']),f(v['lift']),f(v['antecedent_coverage']*100,1)+'%') for v in R['association']['heldout']],[2.6,2.8,3.2,2.8,3.6])
    b.p('Support giữ lại tính trên tất cả transaction của tập tương ứng. Confidence và Lift không xác định khi thiếu vế trái/vế phải được bỏ khỏi trung bình theo đúng hàm đánh giá; cột phủ vế trái cho biết bao nhiêu luật có cơ hội được kiểm tra. Trung bình trên tập con hợp lệ không thể được diễn giải như mọi luật đều tổng quát tốt.')
    b.p('Kết luận theo tiêu chí: Apriori đáp ứng mục tiêu khai phá liên kết trên Train và có thể dùng làm bộ sinh ứng viên kèm lời giải thích. Mức ứng dụng cuối cần dựa trên support/confidence từng luật ở dữ liệu mới, độ phủ vế trái và độ ổn định thời gian. Không suy ra quan hệ nhân quả hay bảo đảm người dùng sẽ thích phim chỉ từ Lift lớn.')


def reduction_results(b):
    b.heading('4.9. THỰC NGHIỆM DIMENSIONALITY REDUCTION')
    b.p('PCA và TruncatedSVD được fit trên cùng vector user–genre chuẩn hóa từ Train. Số components thử là {2,4,8,12,16}; lựa chọn ưu tiên số chiều nhỏ nhất giữ ít nhất 90% phương sai, dùng Validation để đối chiếu lỗi tái tạo. Các vector Validation/Test dựng từ lịch sử giữ lại và biến đổi bằng scaler Train. Phép nén này khác hoàn toàn SVD-CF ở mục hồi quy.')
    b.table('PCA: phương sai giữ lại và lỗi tái tạo',['k','EVR Train','MSE Val','MSE Test'],[
        (r['components'],f(r['explained_variance_ratio']),f(r['reconstruction_error']),f(row(r['algorithm'],group='pca')['reconstruction_error']))
        for r in R['pca']['rows'] if r['split']=='validation' and r['algorithm'].startswith('pca_')],[1.8,4.4,4.4,4.4])
    b.table('TruncatedSVD: phương sai giữ lại và lỗi tái tạo',['k','EVR Train','MSE Val','MSE Test'],[
        (r['components'],f(r['explained_variance_ratio']),f(r['reconstruction_error']),f(row(r['algorithm'],group='pca')['reconstruction_error']))
        for r in R['pca']['rows'] if r['split']=='validation' and r['algorithm'].startswith('truncated_svd_')],[1.8,4.4,4.4,4.4])
    b.figure('reduction_variance','Explained Variance theo components; hai phương pháp gần trùng trên đầu vào đã chuẩn hóa')
    b.figure('reduction_reconstruction','Lỗi tái tạo Validation/Test khi tăng số components')
    b.p('Cả hai phương pháp chọn 12 components theo quy tắc ít chiều nhất giữ ≥90% phương sai. Tại 12 chiều, EVR khoảng 0,9276, MSE Validation khoảng 0,1280 và Test khoảng 0,0873. Cấu hình này đạt mục tiêu phương sai nhưng chưa đồng thời đạt MSE Validation ≤0,10 và ngân sách 8 chiều. Với 16 chiều, lỗi tái tạo giảm rõ nhưng tỷ lệ nén từ 19 chiều thấp hơn.')
    b.p('PCA và TruncatedSVD gần bằng nhau về chất lượng vì cùng nhận dữ liệu chuẩn hóa gần zero-mean; sai khác ở vài chữ số cuối chủ yếu phản ánh solver. Không xếp hạng mô hình chỉ từ sai số số học rất nhỏ. PCA là lựa chọn trình bày/diễn giải gọn hơn trong báo cáo; TruncatedSVD vẫn là phương án tương đương trên biểu diễn này.')
    b.table('Kiểm tra rating downstream và baseline trên cùng Train 70%',['Phương pháp','RMSE Val','RMSE Test','MAE Test'],[
        (name,f(next(r['rmse'] for r in R['pca']['downstream'] if r['algorithm']==a and r['split']=='validation')),
         f(next(r['rmse'] for r in R['pca']['downstream'] if r['algorithm']==a and r['split']=='test')),
         f(next(r['mae'] for r in R['pca']['downstream'] if r['algorithm']==a and r['split']=='test')))
        for a,name in [('pca','PCA 12'),('truncated_svd','SVD 12'),('mean','Mean'),('item_knn','Item-KNN')]],[4.2,3.6,3.6,3.6])
    b.figure('reduction_downstream','Rating từ profile đã tái tạo, đối chiếu Mean và Item-KNN cùng phần fit')
    b.p('Downstream lấy vector user–genre Train sau nén/tái tạo, đảo chuẩn hóa rồi dự đoán rating bằng trung bình các genre của phim; user thiếu hồ sơ dùng mean Train. Mean/Item-KNN trong bảng này cũng chỉ fit Train 70%, nên không trộn với baseline supervised đã refit 85%. Đây là phép kiểm tra tác dụng của biểu diễn, không phải cùng mô hình Ridge của các lần chạy lịch sử.')
    b.p('Trên Validation, Mean/Item-KNN tốt hơn hai profile nén; trên Test, Item-KNN vẫn có RMSE thấp hơn PCA/SVD. Do đó, bảo toàn phương sai chưa tạo lợi thế dự đoán rating ở phép đọc ra đã chọn. Nếu mục tiêu là nén và tái tạo, PCA/SVD đáp ứng một phần; nếu mục tiêu là recommendation, dữ liệu chưa chứng minh nên thay Item-KNN bằng profile giảm chiều.')


def rl_results(b):
    b.heading('4.10. THỰC NGHIỆM REINFORCEMENT LEARNING')
    b.p('Nhóm có một thuật toán học là Q-Learning. Random và Popularity chỉ là mốc so sánh bắt buộc để kiểm tra reward tăng thêm. Môi trường MovieHistoryEnv được cài theo Gymnasium, gồm reset/step, observation_space, action_space và mặt nạ loại phim đã chọn. Train và hai tập giữ lại có phản hồi riêng; Test không tham gia xây Q-table hoặc chọn tham số.')
    b.table('Tham số môi trường và Q-Learning',['Thông số','Giá trị'],[
        ('State','userId; bitmask phim đã chọn; genre trước; phản hồi thích/không thích'),
        ('Action','30 phim phổ biến nhất Train, chưa chọn trong episode'),
        ('Reward có quan sát','(rating − 2,75)/2,25, thuộc [−1;1]'),
        ('Reward không có quan sát','0 theo giả định; không phải rating dự đoán'),
        ('α / γ / ε','0,20 / 0,90 / 0,20'),('Episode huấn luyện','5.000; checkpoint mỗi 250 episode'),
        ('Đánh giá mỗi policy','100 episode × 5 bước; cùng user-episode và seed scenario')],[4.2,10.8])
    b.table('Reward và độ phủ phản hồi của Q-Learning cùng baseline',['Policy / tập','Cumulative','Average/action','Feedback quan sát'],[
        (name+' / '+s.title(),f(row(a,s,'reinforcement')['cumulative_reward'],3),f(row(a,s,'reinforcement')['average_reward']),f(row(a,s,'reinforcement')['observed_feedback_fraction']*100,1)+'%')
        for a,name in [('q_learning','Q-Learning'),('random','Random'),('popular','Popularity')] for s in ('validation','test')],[4.2,3.4,3.7,3.7])
    b.figure('rl_learning_curve','Reward Validation theo episode; Test không được dùng vẽ đường chọn policy')
    b.figure('rl_test_reward_coverage','Reward Test và tỷ lệ hành động có phản hồi lịch sử')
    q=row('q_learning',group='reinforcement');popular=row('popular',group='reinforcement');random=row('random',group='reinforcement')
    b.p(f"Q-Learning đạt Average Reward Test {f(q['average_reward'])}, trong khi mốc tốt nhất trong hai baseline là {f(max(popular['average_reward'],random['average_reward']))}. Chênh lệch Δ={f(q['average_reward']-max(popular['average_reward'],random['average_reward']))} không dương, nên chưa đạt mục tiêu cải thiện reward. Cumulative Reward dương không đủ để kết luận mô hình đạt yêu cầu nếu các baseline còn cao hơn.")
    b.p(f"Độ phủ phản hồi của Q-Learning trên Test chỉ {f(q['observed_feedback_fraction']*100,1)}%; phần còn lại nhận reward 0 theo giả định. Thời gian học khoảng {f(q['fit_seconds'],3)} giây đạt ngân sách nhỏ của mẫu, nhưng tiêu chí hiệu năng không bù cho việc chưa có reward tăng thêm. State chứa userId và lịch sử đã chọn làm Q-table thưa; nhiều user/trạng thái mới theo thời gian khó được khai thác từ bảng đã học.")
    b.p('Kết luận của nhóm được đặt theo tiêu chí: Q-Learning đã được cài đặt và chạy trên môi trường Gymnasium, có log reward, nhưng chưa chứng minh giá trị gợi ý tuần tự hơn baseline. Cần cải tiến khả năng tổng quát hóa state, tăng đa dạng trajectory và dùng dữ liệu logging phù hợp; không đưa Q-Learning vào chế độ mặc định của hệ thống chỉ vì đây là thuật toán học tăng cường.')


def deployment_results(b):
    b.heading('4.11. TỔNG HỢP VÀ TRIỂN KHAI WEB DEMO')
    b.table('Tổng hợp lựa chọn theo đúng bài toán và tiêu chí',['Nhóm','Lựa chọn / đánh giá','Căn cứ và điều kiện'],[
        ('Regression','XGBoost tốt nhất trong 4 mô hình học; Auto vẫn chọn Mean khi có xét baseline','RMSE Validation quyết định; không chọn lại bằng Test; chưa đạt mốc sai số'),
        ('Classification','RF tốt nhất F1 Validation nhưng chưa nên dùng quyết định hiện tại','Ngưỡng 0,10 gần như đoán mọi mẫu là Thích; chưa vượt baseline quyết định'),
        ('Clustering','Ưu tiên K-Means để bao phủ user; chất lượng cụm còn yếu','Silhouette Validation tốt hơn DBSCAN; DBSCAN loại hơn 90% user giữ lại'),
        ('Association','Apriori đạt khai phá Train, dùng làm ứng viên có giải thích','Support/Confidence/Lift và độ phủ vế trái; cần kiểm tra luật giữ lại'),
        ('Reduction','PCA/SVD gần tương đương; 12 chiều đạt EVR, chưa đủ mọi mốc','19→12 chiều; cân nhắc 16 nếu ưu tiên tái tạo; chưa vượt Item-KNN downstream'),
        ('Reinforcement','Q-Learning chưa đạt mục tiêu reward tăng thêm','Δ reward so baseline không dương; coverage thấp, chỉ có đánh giá offline')],[3.0,5.5,6.5])
    b.p('Ứng dụng React/FastAPI hiện có thể hiện luồng danh mục phim, lịch sử rating và gợi ý Top-N. Để không nhầm mô hình đang phục vụ từ các thí nghiệm cũ với bộ số liệu chính, đồ án bổ sung một web demo riêng đọc trực tiếp artifact report-temporal. Demo này không ghi đè con trỏ model của ứng dụng hiện có.')
    b.p('Trong demo báo cáo, Auto chọn mô hình có RMSE Validation thấp nhất khi có xét baseline. Vì Mean thắng ở Validation, Auto dùng điểm trung bình và phá hòa bằng popularity đã học từ Train + Validation; đây là fallback an toàn, chưa cá nhân hóa. Người dùng có thể chọn XGBoost hoặc ứng viên khác để kiểm tra profile fold-in; phim đã đánh giá được loại khỏi Top-N. Không dùng Test để huấn luyện hồ sơ hoặc sắp xếp danh sách.')
    if (ASSETS/'report_demo.png').exists():b.figure('report_demo','Web demo đọc model thật của bộ chạy báo cáo: lịch sử rating và gợi ý Top-N')
    b.p('Kết quả Top-N trong giao diện là minh họa vận hành, không thay một thí nghiệm ranking có Recall@K/NDCG@K và tập ứng viên chuẩn hóa. Báo cáo chưa khẳng định rằng mô hình rating tốt nhất sẽ luôn tạo danh sách gợi ý tốt nhất. Kiểm chứng ranking, đánh giá với user mới và thử nghiệm online được giữ là bước phát triển tiếp theo.')


def conclusion(b):
    b.chapter_title(5,'KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN')
    b.heading('1. KẾT QUẢ ĐẠT ĐƯỢC')
    b.p('Đồ án đã xây dựng quy trình đọc và kiểm tra MovieLens, biểu diễn rating/genre, cài đặt sáu nhóm bài toán, logging artifact và web demo. Bộ chạy chính của báo cáo dùng 49.552 ratings thật, global temporal 70/15/15, năm fold thời gian, GridSearchCV, Optuna cho NCF và môi trường Gymnasium cho Q-Learning. Bảng số liệu, cấu hình và hình vẽ đều truy xuất được đến run ID.')
    b.p('Ở nhóm Regression, XGBoost là mô hình học có RMSE Validation tốt nhất và có RMSE Test thấp nhất trong bốn ứng viên. Tuy nhiên, Mean Rating còn tốt hơn trên Validation nên chưa có bằng chứng đủ để đổi mặc định an toàn sang mô hình học. Linear hạn chế về biểu diễn phi tuyến; SVD và NCF chịu ảnh hưởng mạnh bởi user/phim mới. Các lý giải này là giả thuyết được hỗ trợ bởi dữ liệu bao phủ, không chứng minh kiến trúc nào luôn kém.')
    b.p('Ở nhóm Classification, RF được chọn theo F1 Validation nhưng decision threshold thấp dẫn đến gần như dự đoán toàn bộ là Thích. F1 vượt 0,70 không đồng nghĩa đã vượt baseline; AUC và Accuracy vẫn cho thấy hạn chế. Logistic nhanh hơn nhưng AUC Test thấp hơn RF. Nhóm chưa đủ cơ sở đưa nhãn nhị phân hiện tại thành một quyết định gợi ý đáng tin mà không điều chỉnh mục tiêu và xác nhận lại trên dữ liệu mới.')
    b.p('Ở nhóm Clustering, K-Means phù hợp hơn DBSCAN cho yêu cầu bao phủ vì giữ được mọi user hợp lệ và có Silhouette Validation tốt hơn. Dù vậy, Silhouette/DBI của K-Means chưa đạt mục tiêu đề ra. DBSCAN không được xem là tốt hơn chỉ vì DBI thấp: tỷ lệ noise hơn 90% làm phần giữ lại rất nhỏ và khả năng ứng dụng cho phân đoạn đầy đủ bị hạn chế.')
    b.p('Ở nhóm Association Rules, chỉ có Apriori là thuật toán khai phá chính nên kết luận dựa vào Support, Confidence, Lift, số luật và độ phủ giữ lại. Apriori tìm được các liên kết dương có thể giải thích trên Train. Điều kiện đạt ngưỡng khai phá không chứng minh mọi luật bền vững theo thời gian; chỉ nên dùng làm bộ sinh ứng viên và kiểm tra độ ổn định từng luật.')
    b.p('Ở nhóm Dimensionality Reduction, PCA và TruncatedSVD gần như tương đương trên vector chuẩn hóa. Mười hai chiều đạt mục tiêu giữ 90% phương sai nhưng chưa đồng thời đạt toàn bộ mốc lỗi tái tạo/ngân sách; mười sáu chiều tái tạo tốt hơn nhưng nén ít hơn. Nếu ưu tiên diễn giải và nén tuyến tính, PCA là phương án thuận tiện; nếu ưu tiên rating downstream, kết quả chưa cho thấy lợi thế hơn Item-KNN.')
    b.p('Ở nhóm Reinforcement Learning, Q-Learning đã chạy đúng vòng tương tác và có learning curve, nhưng reward Test chưa vượt mốc baseline. Kết luận là chưa đạt mục tiêu cải thiện, không phải chọn một “thuật toán RL thắng” giữa Q-Learning và các chiến lược không học. Training time nhỏ không bù cho reward tăng thêm chưa có và coverage phản hồi thấp.')
    b.heading('2. ĐÁNH GIÁ MỨC ĐỘ HOÀN THÀNH VÀ HẠN CHẾ')
    b.p('Đề tài đáp ứng các yêu cầu cốt lõi về thuật toán, metric, chia thời gian, CV, tuning có log và demo web. Những phần chưa đủ bằng chứng được giữ nguyên trong kết luận: không huấn luyện toàn bộ 25 triệu ratings, chưa chạy nhiều seed/dataset độc lập, chưa có nested-CV hoặc kiểm định thống kê, chưa có đánh giá online hay causal off-policy evaluation. Báo cáo không quy đổi các phần này thành một tỷ lệ hoàn thành tùy ý.')
    b.p('Mẫu 300 user có lịch sử phân bố không đều theo thời gian nên cold-start cao và sự dịch chuyển lớp đáng kể. Mô hình học từ 70% trước có thể gặp nhiều user/phim mới ở những phần sau. Điều kiện lọc fit, metadata và loss khác nhau giữa các nhóm cũng giới hạn việc diễn giải nguyên nhân. Số liệu thời gian là đo cục bộ theo batch, còn peak RSS không phải bộ nhớ riêng từng model.')
    b.p('Phòng thực nghiệm UI v3 và bộ chạy báo cáo dùng hai giao thức khác nhau. Sự khác biệt đã được công bố để tránh trộn điểm số; việc thống nhất giao thức thành một cấu hình dùng chung cho UI, CLI và báo cáo là công việc kỹ thuật cần tiếp tục. Mọi cập nhật sau khi đọc Test hiện tại cần được coi là giả thuyết mới và xác nhận bằng một khoảng giữ lại mới.')
    b.heading('3. HƯỚNG PHÁT TRIỂN')
    for text in [
        'Mở rộng mẫu và rolling backtest; báo cáo riêng warm user/item, cold-user và cold-item. Giữ Test tương lai mới để xác nhận mọi thay đổi phương pháp.',
        'Bổ sung metadata nội dung, chiến lược cold-start và hiệu chỉnh xác suất; dùng mục tiêu classification có ràng buộc Precision hoặc chi phí FP/FN thay vì tối đa F1 đơn lẻ.',
        'Tăng ngân sách tuning có kiểm soát cho NCF/SVD/XGBoost, thêm ablation đầu vào và loss; đánh giá nhiều seed trước khi khẳng định khác biệt nhỏ.',
        'Đánh giá phân cụm theo độ ổn định, độ bao phủ và mức độ diễn giải; thử biểu diễn loại ảnh hưởng thiên hướng chấm cao/thấp của user.',
        'Kiểm tra luật Apriori trên nhiều cửa sổ thời gian, đo số transaction hỗ trợ và khoảng tin cậy; không chỉ đọc Lift trung bình.',
        'Kiểm tra nén theo tác vụ thực tế: so downstream với baseline mạnh và đánh giá Recall@K/NDCG@K trên cùng tập ứng viên, loại phim đã xem.',
        'Thiết kế state RL có khả năng tổng quát hóa và thu thập logging propensity phù hợp; chỉ đánh giá online sau khi có bằng chứng offline thận trọng.',
        'Hợp nhất cấu hình giao thức, version artifact và lựa chọn mặc định giữa UI/CLI/demo; bổ sung giám sát drift, thời gian và tỷ lệ fallback khi triển khai.'
    ]:b.bullet(text)
    b.p('Kết quả quan trọng nhất của đồ án không phải chứng minh mô hình phức tạp luôn thắng, mà là xây một quy trình cho phép phát hiện khi chúng chưa vượt baseline và lựa chọn đúng theo mục tiêu đã khóa. Một hệ gợi ý đáng tin cần kết hợp chất lượng, khả năng tổng quát hóa, độ bao phủ và tài nguyên với những giới hạn được nêu rõ.')


REFERENCES=[
('[1] GroupLens Research, University of Minnesota','MovieLens 25M Dataset — README','2019','https://files.grouplens.org/datasets/movielens/ml-25m-README.html'),
('[2] F. M. Harper và J. A. Konstan','The MovieLens Datasets: History and Context, ACM TiiS, 5(4), Article 19','2015','https://doi.org/10.1145/2827872'),
('[3] Y. Koren, R. Bell và C. Volinsky','Matrix Factorization Techniques for Recommender Systems, Computer, 42(8), tr. 30–37','2009','https://doi.org/10.1109/MC.2009.263'),
('[4] T. Chen và C. Guestrin','XGBoost: A Scalable Tree Boosting System, KDD, tr. 785–794','2016','https://arxiv.org/abs/1603.02754'),
('[5] X. He, L. Liao, H. Zhang, L. Nie, X. Hu và T.-S. Chua','Neural Collaborative Filtering, WWW, tr. 173–182','2017','https://arxiv.org/abs/1708.05031'),
('[6] L. Breiman','Random Forests, Machine Learning, 45, tr. 5–32','2001','https://doi.org/10.1023/A:1010933404324'),
('[7] M. Ester, H.-P. Kriegel, J. Sander và X. Xu','A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise, KDD, tr. 226–231','1996','https://cdn.aaai.org/KDD/1996/KDD96-037.pdf'),
('[8] R. Agrawal và R. Srikant','Fast Algorithms for Mining Association Rules, VLDB, tr. 487–499','1994','https://www.vldb.org/conf/1994/P487.PDF'),
('[9] R. S. Sutton và A. G. Barto','Reinforcement Learning: An Introduction, ấn bản 2, MIT Press','2018','http://incompleteideas.net/book/the-book-2nd.html'),
('[10] scikit-learn developers','LinearRegression — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html'),
('[11] scikit-learn developers','LogisticRegression — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
('[12] scikit-learn developers','RandomForestClassifier — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
('[13] scikit-learn developers','KMeans — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html'),
('[14] scikit-learn developers','DBSCAN — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html'),
('[15] scikit-learn developers','PCA — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'),
('[16] scikit-learn developers','TruncatedSVD — API documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html'),
('[17] S. Raschka và cộng đồng mlxtend','Apriori và Association Rules — user guide','2026','https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/'),
('[18] scikit-learn developers','GridSearchCV và TimeSeriesSplit — model selection documentation','2026','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html'),
('[19] Optuna contributors','TPESampler — API documentation','2026','https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html'),
('[20] Farama Foundation','Gymnasium — Create a Custom Environment','2026','https://gymnasium.farama.org/introduction/create_custom_env/'),
('[21] PyTorch contributors','Embedding — neural network module documentation','2026','https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html'),
('[22] scikit-learn developers','Model evaluation: quantifying the quality of predictions','2026','https://scikit-learn.org/stable/modules/model_evaluation.html')]


def references_and_appendix(b):
    b.chapter_title(6,'TÀI LIỆU THAM KHẢO')
    b.p('Ngày truy cập các nguồn trực tuyến trong danh mục: 11/11/2026.')
    for author,title,year,url in REFERENCES:
        p=b.p(f'{author}, {title}, {year}. {url}.')
        p.paragraph_format.first_line_indent=Cm(-.8);p.paragraph_format.left_indent=Cm(.8)
    b.chapter_title('A','PHỤ LỤC: TÁI LẬP VÀ DẤU VẾT THỰC NGHIỆM')
    b.heading('A. CÁC LỆNH TÁI LẬP')
    b.p('Thực hiện tại thư mục repository sau khi đã có dữ liệu chuẩn bị. Các thư viện phục vụ báo cáo gồm python-docx, lxml, Matplotlib, Optuna và Gymnasium; bộ kết xuất cập nhật mục lục và PDF có thể dùng LibreOffice headless. Bản báo cáo không cần macro; các phần VBA và ribbon tự chạy không liên quan nội dung được loại khỏi bản phát hành.')
    for command in [
        'project/.venv/Scripts/python.exe -X utf8 project/scripts/report_temporal_experiments.py',
        'project/.venv/Scripts/python.exe -X utf8 project/scripts/report_figures.py',
        'project/.venv/Scripts/python.exe -X utf8 project/scripts/build_project_report.py',
        'powershell -ExecutionPolicy Bypass -File project/scripts/finalize_report.ps1',
        'cd project; .venv/Scripts/python.exe -m uvicorn report_demo:app --app-dir scripts --host 127.0.0.1 --port 8143'
    ]:b.bullet(command)
    b.p('Web demo báo cáo mở tại http://127.0.0.1:8143. Chọn phim đã xem và rating, thêm vào lịch sử, chọn Auto hoặc một mô hình và chạy Top-10. Auto thể hiện quyết định bằng Validation có xét baseline; lựa chọn thủ công chỉ phục vụ khảo sát, không thay lựa chọn được xác nhận trong báo cáo.')
    b.heading('B. RUN ID, DỮ LIỆU VÀ MÃ BĂM')
    b.table('Dấu vết để kiểm tra nguồn số liệu',['Nội dung','Giá trị'],[
        ('Run ID',R['run_id']),('Giao thức',R['provenance']['protocol']),
        ('Mẫu ratings',integer(R['provenance']['sampled_ratings'])),
        ('Fingerprint sample',R['provenance']['sample_sha256']),
        ('Fingerprint prepared manifest',R['provenance']['data_manifest_sha256']),
        ('SHA256 bộ chạy',R['runtime']['script_sha256']),
        ('Artifact chính',f"project/artifacts/report-temporal/{R['run_id']}/results.json"),
        ('Bảng log','metrics.csv trong cùng thư mục'),('Dữ liệu biểu đồ','project/docs/report-assets; PNG/PDF từ artifact')],[4.4,10.6])
    b.p('Các mô hình .joblib, ba split Parquet và dự đoán Test .npy được giữ cùng run. Đồ án không phân phối lại toàn bộ CSV nguồn trong báo cáo. Những số liệu thể hiện thêm chữ số trong JSON là giá trị đo gốc; các bảng chính chỉ làm tròn để trình bày, không tính toán lại quyết định lựa chọn từ số đã làm tròn.')
    b.heading('C. QUY TẮC ĐỌC KẾT QUẢ VÀ KIỂM CHỨNG')
    b.p('Số trang phần mở đầu và nội dung bắt đầu từ 1; các trang trước đó dùng số La Mã. Mục lục, danh mục hình và danh mục bảng được dựng bằng field của Word và cập nhật trước khi xuất PDF. Hai trang bìa giữ nguyên từ file gốc theo yêu cầu để nhóm tự điền thông tin hành chính.')
    b.p('Những kết luận về mô hình chỉ áp dụng cho run được nêu trong phụ lục. Nếu thay dữ liệu, chia tập, threshold hoặc không gian tham số, cần tạo run mới và sinh lại toàn bộ bảng/hình. Không thay số Test vào bảng Validation, không chuyển N/A thành 0 và không đặt lại lựa chọn sau khi nhìn Test hiện tại.')


def build():
    b=Report();front_matter(b);chapter_one(b);chapter_two(b);chapter_three(b);chapter_four(b)
    conclusion(b);references_and_appendix(b);b.save()
    print('Built',STAGING/'Nhom05_DoAn_Sach.docm')
    print('Figures',len(b.inventory['figures']),'Tables',len(b.inventory['tables']))


if __name__=='__main__':build()
