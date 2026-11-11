"""Render the macro-free report through LibreOffice's local UNO bridge.

Executed by a Python version matching the bundled LibreOffice pyuno ABI.
"""

import json
import os
import sys
import time
from pathlib import Path

program=Path(os.environ['REPORT_LO_PROGRAM'])
sys.path.insert(0,str(program))
dll_directory=os.add_dll_directory(str(program))
os.environ.setdefault('URE_BOOTSTRAP',(program/'fundamental.ini').as_uri())
import uno
from com.sun.star.beans import PropertyValue


def prop(name,value):
    p=PropertyValue();p.Name=name;p.Value=value;return p


def render(directory):
    root=Path(directory).resolve()
    local=uno.getComponentContext()
    resolver=local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',local)
    context=None
    for _ in range(60):
        try:
            context=resolver.resolve('uno:socket,host=127.0.0.1,port=20127;urp;StarOffice.ComponentContext')
            break
        except Exception:
            time.sleep(.5)
    if context is None:raise RuntimeError('Local LibreOffice UNO renderer not available')
    desktop=context.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',context)
    never=uno.getConstantByName('com.sun.star.document.MacroExecMode.NEVER_EXECUTE')
    source=uno.systemPathToFileUrl(str(root/'Nhom05_DoAn_Sach.docm'))
    doc=desktop.loadComponentFromURL(source,'_blank',0,(
        prop('Hidden',True),prop('ReadOnly',False),prop('MacroExecutionMode',never),prop('UpdateDocMode',0)))
    try:
        indexes=doc.getDocumentIndexes()
        for _ in range(2):
            doc.TextFields.refresh()
            for i in range(indexes.Count):indexes.getByIndex(i).update()
            doc.refresh()
        print('Updated automatic indexes:',indexes.Count,flush=True)
        # This host was rewriting .docx paths. Keep the OOXML package under a
        # neutral extension until its bytes are sealed in the delivery ZIP.
        target=uno.systemPathToFileUrl(str(root/'Nhom05_DoAn_Final.package'))
        doc.storeToURL(target,(prop('FilterName','Office Open XML Text'),prop('Overwrite',True)))
        pdf=uno.systemPathToFileUrl(str(root/'Nhom05_DoAn_HoanThien.pdf'))
        doc.storeToURL(pdf,(prop('FilterName','writer_pdf_Export'),prop('Overwrite',True)))
        stats={p.Name:p.Value for p in doc.DocumentProperties.DocumentStatistics}
        (root/'libreoffice-validation.json').write_text(json.dumps(
            {'statistics':stats,'indexes':indexes.Count,'macro_execution':'NEVER_EXECUTE'},
            ensure_ascii=False,indent=2),encoding='utf-8')
        print('PDF pages:',stats.get('PageCount'),flush=True)
    finally:
        doc.close(True)
        desktop.terminate()


if __name__=='__main__':render(sys.argv[1])
