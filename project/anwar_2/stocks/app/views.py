from django.shortcuts import render
from .forms import stock_form
from .models import stock

# Create your views here.


def stock_view(request):
    form = stock_form(request.POST, request.FILES)
    if form.is_valid():
        form.save()
        return redirect('app:stock')
    return render(request, 'app/base.html', {'form': stock_form})
