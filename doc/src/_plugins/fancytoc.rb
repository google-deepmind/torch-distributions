module Jekyll
  module FancyToCFilter
    def fancytoc(input)
      converter = Redcarpet::Markdown.new(Redcarpet::Render::HTML_TOC)
      converter.render(input)
    end
  end
end

Liquid::Template.register_filter(Jekyll::FancyToCFilter)
